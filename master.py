from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import math
import multiprocessing
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import psutil

from config import (
    DEFAULT_NLM_H, DEFAULT_NLM_H_COLOR, DEFAULT_NLM_SEARCH_WINDOW,
    DEFAULT_NOISE_SEED, DEFAULT_NOISE_SIGMA, DEFAULT_OVERLAP, DEFAULT_PORT,
    DEFAULT_TILE_SIZE, DISCOVERY_PORT, PROTOCOL_VERSION, SOCKET_TIMEOUT,
)
from image_pipeline import add_gaussian_noise, checksum, process_slice, read_image, save_image
from protocol import recv_bytes, recv_json, send_bytes, send_json


class WorkerBusyError(RuntimeError):
    """Worker explicitly rejected a task because all of its slots are occupied."""


class ProcessResourceMonitor:
    """Sample CPU and system memory while local image processing is running."""

    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.memory_samples: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except psutil.Error:
            pass
        cpu_total = 0.0
        for process in processes:
            try:
                cpu_total += process.cpu_percent(None)
            except psutil.Error:
                continue
        self.cpu_samples.append(cpu_total / max(1, multiprocessing.cpu_count()))
        self.memory_samples.append(psutil.virtual_memory().percent)

    def start(self) -> None:
        self.process.cpu_percent(None)  # prime psutil's CPU counters
        try:
            for child in self.process.children(recursive=True):
                child.cpu_percent(None)
        except psutil.Error:
            pass
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample()

    def stop(self) -> tuple[float, float]:
        self.stop_event.set()
        self.thread.join()
        cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else psutil.virtual_memory().percent
        return cpu, memory


def interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Elapsed wall time covered by intervals, counting overlaps only once."""
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def subtract_intervals(
    intervals: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return interval portions not covered by the occupied intervals."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted((a, b) for a, b in occupied if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    remaining: list[tuple[float, float]] = []
    for start, end in intervals:
        cursor = start
        for block_start, block_end in merged:
            if block_end <= cursor:
                continue
            if block_start >= end:
                break
            if block_start > cursor:
                remaining.append((cursor, min(block_start, end)))
            cursor = max(cursor, block_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append((cursor, end))
    return remaining


def clip_intervals(
    intervals: list[tuple[float, float]],
    windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Keep each measured interval only where it falls inside active windows."""
    return [
        (max(start, window_start), min(end, window_end))
        for start, end in intervals
        for window_start, window_end in windows
        if min(end, window_end) > max(start, window_start)
    ]


def worker_info(host: str, port: int) -> dict:
    with socket.create_connection((host, port), timeout=3) as sock:
        send_json(sock, {"action": "info"})
        return recv_json(sock) | {"ip": host, "port": port}


def discover_workers(timeout: float = 1.5) -> list[dict]:
    found: dict[str, dict] = {}
    broadcasts = {"255.255.255.255"}
    local_networks: set[ipaddress.IPv4Network] = set()
    local_ips: set[str] = set()
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET or not address.address or not address.netmask:
                continue
            try:
                ip = ipaddress.IPv4Address(address.address)
                if ip.is_loopback or ip.is_link_local:
                    continue
                network = ipaddress.IPv4Network(f"{address.address}/{address.netmask}", strict=False)
                broadcasts.add(str(network.broadcast_address))
                local_ips.add(address.address)
                # Keep fallback discovery bounded to a /24 on large networks.
                if network.prefixlen < 24:
                    network = ipaddress.IPv4Network(f"{address.address}/24", strict=False)
                local_networks.add(network)
            except ValueError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", 0))
        for broadcast_ip in sorted(broadcasts):
            try:
                sock.sendto(b"DISCOVER_UPSCAYL_WORKERS_V1", (broadcast_ip, DISCOVERY_PORT))
            except OSError as exc:
                print(f"[DISCOVERY] broadcast {broadcast_ip} failed: {exc}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            try:
                raw, addr = sock.recvfrom(2048)
                info = json.loads(raw.decode())
                if info.get("protocol") == PROTOCOL_VERSION:
                    found[addr[0]] = info | {"ip": addr[0]}
            except socket.timeout:
                break
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[DISCOVERY] invalid response: {exc}")
    def probe(ip: str) -> dict | None:
        if ip in local_ips:
            return None
        try:
            with socket.create_connection((ip, DEFAULT_PORT), timeout=0.2) as conn:
                conn.settimeout(0.5)
                send_json(conn, {"action": "info"})
                info = recv_json(conn)
                if info.get("ok"):
                    return info | {"ip": ip, "port": DEFAULT_PORT}
        except (OSError, ValueError):
            pass
        return None

    targets = [str(ip) for network in local_networks for ip in network.hosts() if str(ip) not in local_ips]
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as pool:
        for worker in pool.map(probe, targets):
            if worker:
                found[worker["ip"]] = worker
    return list(found.values())


def remote_task(
    worker: dict, task: dict, nlm_search_window: int,
    nlm_h: float, nlm_h_color: float, timeout: float = SOCKET_TIMEOUT,
) -> tuple[dict, np.ndarray]:
    image = task["image"]
    payload = np.ascontiguousarray(image).tobytes()
    # Fail quickly when a retry target is offline, while allowing large task
    # payloads/results enough time to cross a congested LAN link.
    with socket.create_connection((worker["ip"], worker["port"]), timeout=min(timeout, 10.0)) as sock:
        sock.settimeout(timeout)
        send_json(sock, {
            "version": PROTOCOL_VERSION, "task_id": task["id"],
            "task_number": task["number"], "task_count": task["count"],
            "master_host": socket.gethostname(),
            "shape": list(image.shape), "operation": "nlm", "nlm_task_schema": 4,
            "nlm_search_window": nlm_search_window,
            "nlm_h": nlm_h, "nlm_h_color": nlm_h_color,
            "checksum": checksum(image),
        })
        acknowledgement = recv_json(sock)
        if acknowledgement.get("busy"):
            raise WorkerBusyError(f"{worker['host']} rejected task: all process slots are busy")
        if not acknowledgement.get("accepted"):
            raise RuntimeError(acknowledgement.get("error", "worker rejected task"))
        upload_started = time.perf_counter()
        send_bytes(sock, payload)
        local_send_seconds = time.perf_counter() - upload_started
        response = recv_json(sock)
        response_received_at = time.perf_counter()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "worker failed"))
        # Measure upload at its receiver. sendall() only measures when local
        # TCP buffers accept bytes, which can be much shorter than delivery.
        reported_upload = response.get("input_receive_seconds")
        upload_seconds = float(reported_upload if reported_upload is not None else local_send_seconds)
        upload_interval = (upload_started, upload_started + upload_seconds)
        download_started = time.perf_counter()
        raw = recv_bytes(sock)
        download_interval = (download_started, time.perf_counter())
        recv_json(sock)  # drain the Worker's completion frame
    result = np.frombuffer(raw, dtype=np.uint8).reshape(response["shape"]).copy()
    if checksum(result) != response["checksum"]:
        raise ValueError("result checksum mismatch")
    compute_seconds = float(response.get("compute_seconds", 0.0))
    response["transport"] = {"upload_interval": upload_interval, "download_interval": download_interval, "compute_interval": (max(upload_interval[1], response_received_at - compute_seconds), response_received_at), "task_interval": (upload_interval[0], download_interval[1])}
    return response, result


def empty_worker_stats(host: str) -> dict:
    return {
        "host": host, "tasks": 0, "failed": 0, "retries": 0,
        "cpu": [], "memory": [],
        "master_upload_intervals": [], "master_download_intervals": [],
        "master_compute_intervals": [],
    }


def local_piece(args: tuple[bytes, tuple[int, ...], int, float, float]) -> np.ndarray:
    raw, shape, nlm_search_window, nlm_h, nlm_h_color = args
    return process_slice(
        np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(),
        nlm_search_window, nlm_h, nlm_h_color,
    )


def make_tile_tasks(image: np.ndarray, tile_size: int, overlap: int) -> tuple[list[dict], int, int]:
    """Build the same row-major tile set for sequential, local, and remote modes."""
    height, width = image.shape[:2]
    columns = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    count = columns * rows
    tasks = []
    for y in range(0, height, tile_size):
        y_end = min(height, y + tile_size)
        context_y0 = max(0, y - overlap)
        context_y1 = min(height, y_end + overlap)
        for x in range(0, width, tile_size):
            x_end = min(width, x + tile_size)
            context_x0 = max(0, x - overlap)
            context_x1 = min(width, x_end + overlap)
            tasks.append({
                "id": uuid.uuid4().hex,
                "label": f"{len(tasks) + 1}/{count}",
                "number": len(tasks) + 1, "count": count,
                "x": x, "x_end": x_end,
                "y": y, "y_end": y_end,
                "context_x0": context_x0, "context_y0": context_y0,
                "image": image[context_y0:context_y1, context_x0:context_x1],
            })
    return tasks, columns, rows


def crop_tile_core(part: np.ndarray, task: dict) -> np.ndarray:
    """Remove the denoising context; NLM keeps input dimensions unchanged."""
    left = task["x"] - task["context_x0"]
    top = task["y"] - task["context_y0"]
    width = task["x_end"] - task["x"]
    height = task["y_end"] - task["y"]
    core = part[top:top + height, left:left + width]
    if core.shape[:2] != (height, width):
        raise ValueError(f"tile core shape mismatch: expected {(height, width)}, got {core.shape[:2]}")
    return core


def stitch_tile_pieces(
    pieces: dict[tuple[int, int], np.ndarray], height: int, width: int,
) -> np.ndarray:
    output = np.empty((height, width, 3), dtype=np.uint8)
    for (tile_y, tile_x), part in pieces.items():
        output[tile_y:tile_y + part.shape[0], tile_x:tile_x + part.shape[1]] = part
    return output


def process_tiles_sequential(
    image: np.ndarray, tile_size: int, overlap: int,
    nlm_search_window: int, nlm_h: float, nlm_h_color: float,
) -> tuple[np.ndarray, int, int]:
    """Run the shared tile workload serially in the current single process."""
    height, width = image.shape[:2]
    tasks, columns, rows = make_tile_tasks(image, tile_size, overlap)
    pieces = {}
    for task in tasks:
        processed = process_slice(task["image"], nlm_search_window, nlm_h, nlm_h_color)
        pieces[(task["y"], task["x"])] = crop_tile_core(processed, task)
    return stitch_tile_pieces(pieces, height, width), columns, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed NLM image denoising coordinator")
    parser.add_argument("--input")
    parser.add_argument("--nlm-search-window", type=int, default=None, help="odd NLM search window; larger values take longer")
    parser.add_argument("--mode", choices=("sequential", "local-parallel", "distributed"))
    parser.add_argument("--worker", action="append", default=[], metavar="HOST[:PORT]")
    parser.add_argument("--workers", type=int, help="limit discovered workers")
    parser.add_argument("--scheduler", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--fault-tolerance", action=argparse.BooleanOptionalAction, default=None, help="retry failed tasks on another Worker")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="core width and height of each square image tile")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    if args.tile_size < 1:
        parser.error("--tile-size must be at least 1")
    if args.nlm_search_window is not None and (args.nlm_search_window < 7 or args.nlm_search_window % 2 == 0):
        parser.error("--nlm-search-window must be an odd integer of at least 7")
    # Keep synthetic input and filter strength consistent across benchmark modes.
    args.noise_sigma = DEFAULT_NOISE_SIGMA
    args.noise_seed = DEFAULT_NOISE_SEED
    args.nlm_h = DEFAULT_NLM_H
    args.nlm_h_color = DEFAULT_NLM_H_COLOR
    workers: list[dict] = []
    planned_worker_count = 0
    # No-argument launch is the guided terminal interface used for demos.
    if not args.input:
        print("=== Distributed NLM Image Denoiser ===")
        while True:
            entered_path = input("Image path or filename in input/: ").strip()
            candidate = Path(entered_path).expanduser()
            if candidate.is_file():
                args.input = str(candidate)
                break
            input_candidate = Path("input") / candidate.name
            if candidate.parent == Path(".") and input_candidate.is_file():
                args.input = str(input_candidate)
                break
            print("File not found. Enter a valid path or a filename located in input/.")
        if Path(args.input).stem.endswith("-before"):
            parser.error("choose the original image in input/, not a generated -before image")
        if args.nlm_search_window is None:
            while True:
                entered = input(f"NLM search window [default {DEFAULT_NLM_SEARCH_WINDOW}, odd >= 7]: ").strip()
                if not entered:
                    args.nlm_search_window = DEFAULT_NLM_SEARCH_WINDOW
                    break
                try:
                    candidate_window = int(entered)
                    if candidate_window >= 7 and candidate_window % 2 == 1:
                        args.nlm_search_window = candidate_window
                        break
                except ValueError:
                    pass
                print("Enter an odd integer of at least 7.")
        print("\nChoose execution mode:\n  1) Sequential\n  2) Local parallel\n  3) Distributed")
        modes = {"1": "sequential", "2": "local-parallel", "3": "distributed"}
        while args.mode not in modes.values(): args.mode = modes.get(input("Mode [1-3]: ").strip())
        if args.mode == "distributed":
            print("[MASTER] Scanning LAN for Workers...")
            found = discover_workers()
            if not found:
                print("  No Workers discovered automatically.")
                print("Enter Worker addresses as host:port separated by commas (blank to cancel).")
                raw = input("Workers: ").strip()
                for spec in filter(None, (item.strip() for item in raw.split(","))):
                    host, _, port = spec.partition(":")
                    try: found.append(worker_info(host, int(port or DEFAULT_PORT)))
                    except OSError as exc: print(f"Cannot reach {spec}: {exc}")
            for i, worker in enumerate(found, 1):
                print(f"  [{i}] {worker['host']} ({worker['ip']}, {worker.get('slots', worker.get('cores', 1))} processes)")
            if found:
                selection = input("Workers to use [all, or numbers like 1,2,3]: ").strip().lower()
                if selection not in ("", "all", "*"):
                    try: found = [found[int(item)-1] for item in selection.split(",") if 0 < int(item) <= len(found)]
                    except ValueError: print("Invalid selection; using all discovered Workers.")
            args.worker = [f"{w['ip']}:{w['port']}" for w in found]
            workers = found
            print("Scheduler:\n  1) Dynamic\n  2) Static")
            args.scheduler = "static" if input("Scheduler [1-2, default 1]: ").strip() == "2" else "dynamic"
            args.fault_tolerance = input("Enable fault tolerance? [y/N]: ").strip().lower() in ("y", "yes")
        print()
    if args.nlm_search_window is None:
        args.nlm_search_window = DEFAULT_NLM_SEARCH_WINDOW
    tile_overlap = max(DEFAULT_OVERLAP, args.nlm_search_window // 2 + 7 // 2)
    args.mode = args.mode or "distributed"
    # Discovery is setup and must not be included in T_run.
    if args.mode == "distributed":
        if not workers:
            workers = discover_workers()
        requested_hosts = {spec.partition(":")[0] for spec in args.worker}
        if requested_hosts:
            workers = [w for w in workers if w["ip"] in requested_hosts]
        for spec in args.worker:
            host, _, port = spec.partition(":")
            if host not in {w["ip"] for w in workers}:
                workers.append(worker_info(host, int(port or DEFAULT_PORT)))
        if args.workers:
            workers = workers[:args.workers]
        if not workers:
            parser.error("no Workers discovered; start worker.py before launching Master")
        planned_worker_count = len(workers)
        all_workers = list(workers)
        unsupported = [
            w for w in workers
            if "nlm" not in w.get("operations", []) or w.get("nlm_task_schema") != 4
        ]
        if unsupported:
            names = ", ".join(f"{w['host']}@{w['ip']}" for w in unsupported)
            parser.error(f"Workers {names} have an incompatible task format; copy the latest worker.py and image_pipeline.py there, then restart them")
        print("[MASTER] Selected Workers: " + ", ".join(f"{w['host']}@{w['ip']}" for w in workers))
    source_path = Path(args.input).expanduser()
    if not source_path.is_file() and source_path.parent == Path("."):
        source_path = Path("input") / source_path.name
    if not source_path.is_file():
        parser.error(f"image not found: {args.input}; choose a file in input/ or provide its path")
    if source_path.stem.endswith("-before"):
        parser.error("choose the original image, not a generated -before image")
    original_path = source_path
    original_image = read_image(original_path)
    noise_preparation_started = time.perf_counter()
    image = add_gaussian_noise(original_image, args.noise_sigma, args.noise_seed)
    before_path = Path("input") / f"{original_path.stem}-before.png"
    save_image(before_path, image)
    noise_preparation_seconds = time.perf_counter() - noise_preparation_started
    args.input = str(before_path)
    h, w, _ = image.shape
    print(
        f"[INPUT] original={original_path} noisy={before_path} {w}x{h} "
        f"noise_prep={noise_preparation_seconds:.3f}s"
    )
    print(f"[MASTER] NLM search_window={args.nlm_search_window} mode={args.mode}")
    baseline_seconds = None
    baseline_resource_usage = None
    if args.mode != "sequential":
        print("[BENCHMARK] Measuring single-process baseline...")
        baseline_monitor = ProcessResourceMonitor()
        baseline_monitor.start()
        baseline_started = time.perf_counter()
        sequential_result, baseline_columns, baseline_rows = process_tiles_sequential(
            image, args.tile_size, tile_overlap, args.nlm_search_window,
            args.nlm_h, args.nlm_h_color,
        )
        baseline_seconds = time.perf_counter() - baseline_started
        baseline_resource_usage = baseline_monitor.stop()
        print(
            f"[BENCHMARK] sequential={baseline_seconds:.3f}s "
            f"grid={baseline_columns}x{baseline_rows} tasks={baseline_columns * baseline_rows} "
            f"checksum={checksum(sequential_result)}"
        )
    worker_stats: dict[str, dict] = {}
    task_details: list[dict] = []
    failure_events: list[str] = []
    recovery_seconds = 0.0
    failed_tasks: list[dict] = []
    tile_columns = tile_rows = task_count = 0
    started = time.perf_counter()
    local_resource_monitor = None
    if args.mode != "distributed":
        local_resource_monitor = ProcessResourceMonitor()
        local_resource_monitor.start()
    task_preparation_seconds = 0.0
    parallel_wall_seconds = 0.0
    stitching_seconds = 0.0
    processing_interval = (started, started)
    if args.mode == "sequential":
        preparation_started = time.perf_counter()
        tasks, tile_columns, tile_rows = make_tile_tasks(image, args.tile_size, tile_overlap)
        task_count = len(tasks)
        process_slots = 1
        task_preparation_seconds = time.perf_counter() - preparation_started
        print(
            f"[TASK DECOMPOSITION] core_tile={args.tile_size}x{args.tile_size} "
            f"grid={tile_columns}x{tile_rows} task_count={task_count} "
            f"context={tile_overlap}px process_slots=1"
        )
        pieces = {}
        processing_started = time.perf_counter()
        for task in tasks:
            processed = process_slice(
                task["image"], args.nlm_search_window, args.nlm_h, args.nlm_h_color
            )
            pieces[(task["y"], task["x"])] = crop_tile_core(processed, task)
        processing_interval = (processing_started, time.perf_counter())
        parallel_wall_seconds = processing_interval[1] - processing_interval[0]
        result = stitch_tile_pieces(pieces, h, w)
    else:
        preparation_started = time.perf_counter()
        process_slots = (sum(max(1, int(worker.get("slots", worker.get("cores", 1)))) for worker in workers)
                         if args.mode == "distributed" else (multiprocessing.cpu_count() or 1))
        tasks, tile_columns, tile_rows = make_tile_tasks(image, args.tile_size, tile_overlap)
        task_count = tile_columns * tile_rows
        print(
            f"[TASK DECOMPOSITION] core_tile={args.tile_size}x{args.tile_size} "
            f"grid={tile_columns}x{tile_rows} task_count={task_count} "
            f"context={tile_overlap}px process_slots={process_slots}"
        )
        task_preparation_seconds = time.perf_counter() - preparation_started
        pieces: dict[tuple[int, int], np.ndarray] = {}
        parallel_started = time.perf_counter()
        if args.mode == "local-parallel":
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_slots) as pool:
                futures = {
                    pool.submit(
                        local_piece,
                        (t["image"].tobytes(), t["image"].shape, args.nlm_search_window, args.nlm_h, args.nlm_h_color),
                    ): t for t in tasks
                }
                for f, t in futures.items():
                    pieces[(t["y"], t["x"])] = crop_tile_core(f.result(), t)
        else:
            print("[MASTER] workers=" + ", ".join(f"{w['host']}@{w['ip']}({w.get('slots', w.get('cores', 1))} processes)" for w in workers))
            pending_tasks = list(tasks)
            # Give every distinct Worker one initial task before assigning its
            # additional local process slots. This prevents the first node in
            # the list from consuming a small workload by itself.
            slots = []
            max_slots = max((max(1, int(w.get("slots", w.get("cores", 1)))) for w in workers), default=1)
            for slot_index in range(max_slots):
                for worker_node in workers:
                    worker_capacity = max(1, int(worker_node.get("slots", worker_node.get("cores", 1))))
                    if slot_index < worker_capacity:
                        slots.append(worker_node)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
                active = {}
                static_queues: list[list[dict]] = []
                def submit(worker, task, slot_index=None):
                    task.setdefault("assigned_worker", worker["host"])
                    task.setdefault("assigned_worker_ip", worker["ip"])
                    active[pool.submit(
                        remote_task, worker, task, args.nlm_search_window, args.nlm_h, args.nlm_h_color
                    )] = (worker, task, slot_index)
                if args.scheduler == "static":
                    assignments = [[] for _ in slots]
                    for index, task in enumerate(pending_tasks): assignments[index % len(slots)].append(task)
                    pending_tasks.clear()
                    static_queues = assignments
                    # A static assignment fixes each task to a Worker slot,
                    # but only one in-flight task is allowed per slot. Further
                    # tasks on that slot wait until its current task returns.
                    for slot_index, group in enumerate(static_queues):
                        if group:
                            submit(slots[slot_index], group.pop(0), slot_index)
                else:
                    for worker in slots:
                        if pending_tasks: submit(worker, pending_tasks.pop(0))
                while active:
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        worker, task, slot_index = active.pop(future)
                        retried = False
                        lost = False
                        first_error = None
                        try:
                            response, part = future.result()
                        except WorkerBusyError as busy_error:
                            first_error = busy_error
                            failed_worker = worker
                        except Exception as error:
                            first_error = error
                            failed_worker = worker
                        if first_error is not None:
                            task["failed_on"] = failed_worker["host"]
                            failed_stats = worker_stats.setdefault(
                                failed_worker["ip"], empty_worker_stats(failed_worker["host"])
                            )
                            failed_stats["failed"] += 1
                            event = "WORKER BUSY" if isinstance(first_error, WorkerBusyError) else "TASK FAILED"
                            failure_line = f"[{event}] task={task['label']} worker={failed_worker['host']}@{failed_worker['ip']} reason={type(first_error).__name__}: {first_error}"
                            print(failure_line); failure_events.append(failure_line)
                            if not args.fault_tolerance:
                                if args.scheduler == "dynamic":
                                    workers = [w for w in workers if w["ip"] != failed_worker["ip"]]
                                    slots = [w for w in slots if w["ip"] != failed_worker["ip"]]
                                task["status"] = "FAILED"
                                failed_tasks.append(task); lost = True
                                print(f"[PARTIAL] task={task['label']} marked missing; its output region will be highlighted")
                            else:
                                recovery_started = time.perf_counter()
                                recovered = False
                                # Keep temporarily unavailable Workers eligible. A busy response
                                # means capacity is occupied, not that the Worker has failed.
                                retry_candidates = [w for w in all_workers if w["ip"] != failed_worker["ip"]]
                                retry_candidates += [w for w in all_workers if w["ip"] == failed_worker["ip"]]
                                for retry_round in range(1, 5):
                                    saw_busy = False
                                    for candidate in retry_candidates:
                                        retry_line = f"[RETRY] task={task['label']} attempt={retry_round} target={candidate['host']}@{candidate['ip']}"
                                        print(retry_line); failure_events.append(retry_line)
                                        try:
                                            response, part = remote_task(
                                                candidate, task, args.nlm_search_window, args.nlm_h, args.nlm_h_color
                                            )
                                            worker = candidate; recovered = True; retried = True
                                            break
                                        except WorkerBusyError as retry_error:
                                            saw_busy = True
                                            busy_line = f"[RETRY WAIT] task={task['label']} worker={candidate['host']} is busy; will try again"
                                            print(busy_line); failure_events.append(busy_line)
                                        except Exception as retry_error:
                                            retry_error_line = f"[RETRY FAILED] task={task['label']} worker={candidate['host']} reason={type(retry_error).__name__}: {retry_error}"
                                            print(retry_error_line); failure_events.append(retry_error_line)
                                    if recovered:
                                        break
                                    if retry_round < 4:
                                        delay = min(2 ** (retry_round - 1), 8)
                                        wait_line = f"[RETRY BACKOFF] task={task['label']} round={retry_round} wait={delay}s" + (" (Workers busy)" if saw_busy else "")
                                        print(wait_line); failure_events.append(wait_line)
                                        time.sleep(delay)
                                if not recovered:
                                    task["status"] = "FAILED"
                                    failed_tasks.append(task); lost = True
                                    print(f"[PARTIAL] task={task['label']} retry limit reached; output region will be highlighted")
                                else:
                                    task_recovery = time.perf_counter() - recovery_started
                                    recovery_seconds += task_recovery
                                    print(f"[RECOVERED] task={task['label']} completed_on={worker['host']} recovery={task_recovery:.3f}s")
                        if not lost:
                            transport = response.get("transport", {})
                            task_window = [tuple(transport.get("task_interval", (0, 0)))]
                            task_network_intervals = clip_intervals(
                                [tuple(transport.get("upload_interval", (0, 0))),
                                 tuple(transport.get("download_interval", (0, 0)))],
                                task_window,
                            )
                            task_compute_intervals = clip_intervals(
                                [tuple(transport.get("compute_interval", (0, 0)))],
                                task_window,
                            )
                            task_compute = interval_union_seconds(task_compute_intervals)
                            stats = worker_stats.setdefault(worker["ip"], empty_worker_stats(worker["host"]))
                            stats["tasks"] += 1
                            stats["master_upload_intervals"].extend(clip_intervals(
                                [tuple(transport.get("upload_interval", (0, 0)))], task_window
                            ))
                            stats["master_download_intervals"].extend(clip_intervals(
                                [tuple(transport.get("download_interval", (0, 0)))], task_window
                            ))
                            stats["master_compute_intervals"].extend(task_compute_intervals)
                            stats["cpu"].append(float(response.get("cpu_percent", 0.0)))
                            stats["memory"].append(float(response.get("memory_percent", 0.0)))
                            stats["retries"] += int(retried)
                            if retried and worker["ip"] != task.get("assigned_worker_ip"):
                                task_status = f"REROUTED TO {worker['host']}"
                            elif retried:
                                task_status = f"RETRIED ON {worker['host']}"
                            else:
                                task_status = "SUCCESS"
                            task["status"] = task_status
                            region = f"x={task['x']}..{task['x_end']-1},y={task['y']}..{task['y_end']-1}"
                            detail = {"id": task["id"], "label": task["label"], "worker": task.get("assigned_worker", worker["host"]), "status": task_status, "region": region}
                            detail["network_intervals"] = task_network_intervals
                            detail["compute"] = task_compute
                            task_details.append(detail)
                            print(f"[TASK] {task['label']} -> {worker['host']}")
                            pieces[(task["y"], task["x"])] = crop_tile_core(part, task)
                        if args.scheduler == "static" and slot_index is not None and static_queues[slot_index]:
                            submit(slots[slot_index], static_queues[slot_index].pop(0), slot_index)
                        if args.scheduler == "dynamic" and pending_tasks and slots:
                            submit(worker if worker in slots else slots[0], pending_tasks.pop(0))
                if pending_tasks:
                    if args.fault_tolerance:
                        raise RuntimeError("no active Worker slots for pending tasks")
                    for task in pending_tasks:
                        task["status"] = "FAILED (NO WORKER)"
                    failed_tasks.extend(pending_tasks)
                    print(f"[PARTIAL] {len(pending_tasks)} pending task(s) have no available Worker and will be highlighted")
                    pending_tasks.clear()
        parallel_wall_seconds = time.perf_counter() - parallel_started
        processing_interval = (parallel_started, parallel_started + parallel_wall_seconds)
        stitching_started = time.perf_counter()
        if failed_tasks:
            result = np.full((h, w, 3), (255, 0, 255), dtype=np.uint8)
            for (task_y, task_x), part in pieces.items():
                result[task_y:task_y + part.shape[0], task_x:task_x + part.shape[1]] = part
            print(f"[PARTIAL IMAGE] {len(failed_tasks)} task region(s) highlighted magenta.")
        else:
            result = stitch_tile_pieces(pieces, h, w)
        stitching_seconds = time.perf_counter() - stitching_started
    elapsed = time.perf_counter() - started
    local_resource_usage = None
    if local_resource_monitor is not None:
        local_resource_usage = local_resource_monitor.stop()
    # In sequential mode this run is the baseline itself. Show its measured
    # duration and the expected 1x / 100% single-process reference values.
    if args.mode == "sequential":
        baseline_seconds = elapsed
    path = Path(args.output) / f"{original_path.stem}-after-nlm-sw{args.nlm_search_window}.png"
    save_image(path, result)
    status = "PARTIAL" if failed_tasks else "COMPLETE"
    print(f"[MASTER] status={status} output={path} shape={result.shape} checksum={checksum(result)} elapsed={elapsed:.3f}s")
    throughput = h * w / elapsed if elapsed else 0.0
    report = [
        "Distributed NLM Image Denoising - Benchmark Report",
        f"Master: {socket.gethostname()}",
        f"Original image: {original_path}",
        f"Noisy input: {before_path}",
        f"Resolution: {w}x{h}",
        f"Noise preparation: {noise_preparation_seconds:.6f}s (excluded from T_run)",
        f"NLM search window: {args.nlm_search_window}",
        f"Mode: {args.mode}", f"Scheduler: {args.scheduler}", f"Status: {status}",
        f"Fault tolerance: {'Y' if args.fault_tolerance else 'N' if args.fault_tolerance is not None else 'n/a'}",
        f"Output: {path}", f"Execution time: {elapsed:.6f}s",
        f"Throughput: {throughput:.3f} input-pixels/s",
    ]
    parallelism = process_slots
    speedup = baseline_seconds / elapsed if baseline_seconds is not None and elapsed else None
    efficiency = speedup / max(1, parallelism) * 100 if speedup is not None else None
    analysis_rows = [
        ("Operation", "NLM denoising"),
        ("Input", f"{original_path.name} -> {before_path.name}"),
        ("Noise preparation", f"{noise_preparation_seconds:.3f}s; excluded from T_run"),
        ("NLM search window", str(args.nlm_search_window)),
        ("Execution mode", args.mode),
        ("Scheduler", args.scheduler),
        ("Fault tolerance", "Y" if args.fault_tolerance else "N" if args.fault_tolerance is not None else "n/a"),
        ("Worker nodes", str(planned_worker_count)),
        ("Process slots P", str(parallelism)),
        ("T_seq", f"{baseline_seconds:.3f}s" if baseline_seconds is not None else "baseline run"),
        ("T_run", f"{elapsed:.3f}s"),
        ("Speedup", f"{speedup:.3f}x" if speedup is not None else "n/a"),
        ("Efficiency", f"{efficiency:.2f}% (P={parallelism})" if efficiency is not None else "n/a"),
        ("Throughput", f"{throughput:.2f} input-pixels/s"),
    ]
    print("\n[OVERALL ANALYSIS]")
    print(f"{'METRIC':<24} VALUE")
    report.extend(["", "OVERALL ANALYSIS", f"{'METRIC':<24} VALUE"])
    for label, value in analysis_rows:
        line = f"{label:<24} {value}"
        print(line); report.append(line)
    if args.mode == "distributed":
        all_network_intervals = [interval for stat in worker_stats.values() for interval in (stat["master_upload_intervals"] + stat["master_download_intervals"])]
        all_compute_intervals = [interval for stat in worker_stats.values() for interval in stat["master_compute_intervals"]]
        run_window = [(parallel_started, parallel_started + parallel_wall_seconds)]
        run_network = clip_intervals(all_network_intervals, run_window)
        run_compute = clip_intervals(all_compute_intervals, run_window)
        all_compute = interval_union_seconds(run_compute)
        # Count network/compute overlap as compute time, not network time.
        run_network = subtract_intervals(run_network, run_compute)
        all_network = interval_union_seconds(run_network)
        processing_label = "Pure compute (all Workers)"
    else:
        run_window = [(started, started + elapsed)]
        run_compute = clip_intervals([processing_interval], run_window)
        all_compute = interval_union_seconds(run_compute)
        all_network = 0.0
        processing_label = "Sequential processing" if args.mode == "sequential" else "Local parallel processing"
    # Residual wall time includes task preparation, stitching, and runtime overhead.
    run_other = max(0.0, elapsed - all_network - all_compute)
    phase_rows = [
        ("Network (all Workers)" if args.mode == "distributed" else "Network", all_network),
        (processing_label, all_compute),
        ("Other (whole job)", run_other),
        ("T_run total", elapsed),
    ]
    print("\n[RUN TIME OVERALL]")
    phase_header = f"{'PHASE':<30} {'WALL TIME':>12}"
    print(phase_header)
    report.extend(["", "RUN TIME OVERALL", phase_header])
    for phase, seconds in phase_rows:
        line = f"{phase:<30} {seconds:>11.3f}s"
        print(line); report.append(line)
    master_rows = []
    if args.mode == "sequential":
        cpu, memory = local_resource_usage or (0.0, psutil.virtual_memory().percent)
        master_rows.append(("localhost-seq", task_count, elapsed, cpu, memory))
    else:
        cpu, memory = baseline_resource_usage or (0.0, psutil.virtual_memory().percent)
        master_rows.append(("localhost-seq", baseline_columns * baseline_rows, baseline_seconds, cpu, memory))
        if args.mode == "local-parallel":
            cpu, memory = local_resource_usage or (0.0, psutil.virtual_memory().percent)
            master_rows.append(("localhost-multi", task_count, elapsed, cpu, memory))
    master_header = f"{'HOST':<22} {'TASKS':>7} {'T_RUN':>10} {'AVG CPU':>9} {'AVG RAM':>9}"
    print("\n[MASTER TIME]")
    print(master_header)
    report.extend(["", "MASTER TIME", master_header])
    for host, tasks_done, run_seconds, avg_cpu, avg_ram in master_rows:
        row = f"{host:<22} {tasks_done:>7} {run_seconds:>9.3f}s {avg_cpu:>8.1f}% {avg_ram:>8.1f}%"
        print(row); report.append(row)
    if worker_stats:
        header = f"{'WORKER':<22} {'TASKS':>5} {'FAILED':>6} {'NETWORK':>10} {'COMPUTE':>9} {'AVG CPU':>9} {'AVG RAM':>9} {'RETRY':>6}"
        print("\n[WORKER TIME]")
        print(header)
        report.extend(["", "WORKER PERFORMANCE", header])
        for stat in worker_stats.values():
            avg_cpu = sum(stat["cpu"]) / len(stat["cpu"]) if stat["cpu"] else 0.0
            avg_mem = sum(stat["memory"]) / len(stat["memory"]) if stat["memory"] else 0.0
            worker_network = clip_intervals(
                stat["master_upload_intervals"] + stat["master_download_intervals"],
                run_window,
            )
            worker_compute = clip_intervals(stat["master_compute_intervals"], run_window)
            # Use the same job-wide compute overlap removed from T_RUN.
            worker_network = subtract_intervals(worker_network, run_compute)
            network_wall = interval_union_seconds(worker_network)
            compute_wall = interval_union_seconds(worker_compute)
            row = f"{stat['host'][:22]:<22} {stat['tasks']:>5} {stat['failed']:>6} {network_wall:>9.3f}s {compute_wall:>8.3f}s {avg_cpu:>8.1f}% {avg_mem:>8.1f}% {stat['retries']:>6}"
            print(row); report.append(row)
    detailed_ids = {item["id"] for item in task_details}
    for task in failed_tasks:
        if task["id"] in detailed_ids:
            continue
        task_details.append({
            "id": task["id"],
            "label": task["label"],
            "worker": task.get("assigned_worker", task.get("failed_on", "-")),
            "status": task.get("status", "FAILED"),
            "region": f"x={task['x']}..{task['x_end']-1},y={task['y']}..{task['y_end']-1}",
            "network": None,
            "compute": None,
        })
    task_header = f"{'TASK':<12} {'WORKER':<22} {'REGION':<27} {'STATUS':<30} {'NETWORK':>10} {'COMPUTE':>9}"
    print("\n[TASK TIME]"); print(task_header)
    report.extend(["", "TASK TIME", task_header])
    for item in task_details:
        if item.get("compute") is None:
            network = compute = "-"
        else:
            # Keep task Network on the same basis as the global T_RUN row:
            # any concurrent Worker compute interval is excluded.
            item["network"] = interval_union_seconds(
                subtract_intervals(item["network_intervals"], run_compute)
            )
            network = f"{item['network']:.3f}s"
            compute = f"{item['compute']:.3f}s"
        row = f"{item['label']:<12} {item['worker'][:22]:<22} {item['region']:<27} {item['status']:<30} {network:>10} {compute:>9}"
        print(row); report.append(row)
    recovery_line = f"[RECOVERY] retries={sum(x['retries'] for x in worker_stats.values())} recovery_time={recovery_seconds:.3f}s"
    print("\n" + recovery_line); report.append(recovery_line)
    report.extend(failure_events)
    log_dir = Path("logs"); log_dir.mkdir(parents=True, exist_ok=True)
    safe_master = "".join(c if c.isalnum() or c in "-_" else "_" for c in socket.gethostname())
    log_name = f"{safe_master}_{w}x{h}_nlm_sw{args.nlm_search_window}_{args.mode}_{args.scheduler}_{len(workers) if args.mode == 'distributed' else 0}workers_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    log_path = log_dir / log_name
    log_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[LOG] benchmark report saved: {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
