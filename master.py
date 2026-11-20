from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import multiprocessing
import socket
import time
import uuid
from pathlib import Path

import numpy as np
import psutil

from config import DEFAULT_OVERLAP, DEFAULT_PORT, DEFAULT_TILE_SIZE, DISCOVERY_PORT, MAX_SCALE, MIN_SCALE, PROTOCOL_VERSION, SOCKET_TIMEOUT
from image_pipeline import checksum, process_slice, read_image, save_image
from protocol import recv_bytes, recv_json, send_bytes, send_json


class WorkerBusyError(RuntimeError):
    """Worker explicitly rejected a task because all of its slots are occupied."""


def scale_arg(value: str) -> float:
    scale = float(value)
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise argparse.ArgumentTypeError("scale must be between 1 and 10")
    return scale


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
    worker: dict, task: dict, scale: float, operation: str,
    nlm_search_window: int, timeout: float = SOCKET_TIMEOUT,
) -> tuple[dict, np.ndarray]:
    image = task["image"]
    task_started = time.perf_counter()
    connect_started = time.perf_counter()
    connect_seconds = upload_seconds = 0.0
    payload = np.ascontiguousarray(image).tobytes()
    # Fail quickly when a retry target is offline, while allowing large task
    # payloads/results enough time to cross a congested LAN link.
    with socket.create_connection((worker["ip"], worker["port"]), timeout=min(timeout, 10.0)) as sock:
        connect_seconds = time.perf_counter() - connect_started
        sock.settimeout(timeout)
        send_json(sock, {
            "version": PROTOCOL_VERSION, "task_id": task["id"],
            "shape": list(image.shape), "scale": scale,
            "operation": operation, "nlm_search_window": nlm_search_window,
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
        response_started = time.perf_counter()
        response = recv_json(sock)
        response_received_at = time.perf_counter()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "worker failed"))
        # Measure upload at its receiver. sendall() only measures when local
        # TCP buffers accept bytes, which can be much shorter than delivery.
        input_interval = response.get("input_receive_interval", (0.0, 0.0))
        reported_upload = response.get("input_receive_seconds")
        if reported_upload is None and len(input_interval) == 2 and input_interval[1] > input_interval[0]:
            reported_upload = input_interval[1] - input_interval[0]
        upload_seconds = float(reported_upload if reported_upload is not None else local_send_seconds)
        upload_interval = (upload_started, upload_started + upload_seconds)
        response_wait_seconds = response_received_at - response_started
        download_started = time.perf_counter()
        raw = recv_bytes(sock)
        download_seconds = time.perf_counter() - download_started
        download_interval = (download_started, time.perf_counter())
        phase_intervals = recv_json(sock)
    result = np.frombuffer(raw, dtype=np.uint8).reshape(response["shape"]).copy()
    if checksum(result) != response["checksum"]:
        raise ValueError("result checksum mismatch")
    compute_seconds = float(response.get("compute_seconds", 0.0))
    response["transport"] = {"connect": connect_seconds, "upload": upload_seconds, "upload_interval": upload_interval, "busy_wait": 0.0, "worker_wait": response_wait_seconds, "download": download_seconds, "download_interval": download_interval, "compute_interval": (max(upload_interval[1], response_received_at - compute_seconds), response_received_at), "task_interval": (upload_interval[0], download_interval[1]), "total": time.perf_counter() - task_started, "sent_bytes": len(payload), "received_bytes": len(raw)}
    response["phase_intervals"] = phase_intervals
    return response, result


def empty_worker_stats(host: str) -> dict:
    return {
        "host": host, "tasks": 0, "failed": 0, "compute": 0.0,
        "task_total": 0.0, "other": 0.0, "worker_elapsed": 0.0,
        "cpu": [], "memory": [], "bytes": 0, "retries": 0,
        "connect": 0.0, "upload": 0.0, "busy_wait": 0.0,
        "worker_wait": 0.0, "download": 0.0,
        "network_bytes_in": 0, "network_bytes_out": 0,
        "compute_intervals": [], "upload_intervals": [],
        "download_intervals": [], "active_intervals": [],
        "master_upload_intervals": [], "master_download_intervals": [],
        "master_compute_intervals": [], "master_active_intervals": [],
    }


def local_piece(args: tuple[bytes, tuple[int, ...], float, str, int]) -> np.ndarray:
    raw, shape, scale, operation, nlm_search_window = args
    return process_slice(
        np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(),
        scale, operation, nlm_search_window,
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
                "x": x, "x_end": x_end,
                "y": y, "y_end": y_end,
                "context_x0": context_x0, "context_y0": context_y0,
                "image": image[context_y0:context_y1, context_x0:context_x1],
            })
    return tasks, columns, rows


def crop_tile_core(part: np.ndarray, task: dict, scale: float) -> np.ndarray:
    """Crop context from a processed tile using global output coordinates."""
    out_x0 = round(task["x"] * scale)
    out_x1 = round(task["x_end"] * scale)
    out_y0 = round(task["y"] * scale)
    out_y1 = round(task["y_end"] * scale)
    context_out_x0 = round(task["context_x0"] * scale)
    context_out_y0 = round(task["context_y0"] * scale)
    expected_width = out_x1 - out_x0
    expected_height = out_y1 - out_y0
    left = out_x0 - context_out_x0
    top = out_y0 - context_out_y0
    right = left + expected_width
    bottom = top + expected_height

    # cv2.resize rounds each tile's output dimensions independently. Correct a
    # possible one-pixel rounding difference at the outer image boundaries.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > part.shape[1]:
        left -= right - part.shape[1]
        right = part.shape[1]
    if bottom > part.shape[0]:
        top -= bottom - part.shape[0]
        bottom = part.shape[0]

    core = part[top:bottom, left:right]
    if core.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"tile core shape mismatch: expected {(expected_height, expected_width)}, got {core.shape[:2]}"
        )
    return core


def stitch_tile_pieces(
    pieces: dict[tuple[int, int], np.ndarray], height: int, width: int, scale: float,
) -> np.ndarray:
    output = np.empty((round(height * scale), round(width * scale), 3), dtype=np.uint8)
    for (tile_y, tile_x), part in pieces.items():
        out_y = round(tile_y * scale)
        out_x = round(tile_x * scale)
        output[out_y:out_y + part.shape[0], out_x:out_x + part.shape[1]] = part
    return output


def process_tiles_sequential(
    image: np.ndarray, scale: float, tile_size: int, overlap: int,
    operation: str, nlm_search_window: int,
) -> tuple[np.ndarray, int, int]:
    """Run the shared tile workload serially in the current single process."""
    height, width = image.shape[:2]
    tasks, columns, rows = make_tile_tasks(image, tile_size, overlap)
    pieces = {}
    for task in tasks:
        processed = process_slice(task["image"], scale, operation, nlm_search_window)
        pieces[(task["y"], task["x"])] = crop_tile_core(processed, task, scale)
    return stitch_tile_pieces(pieces, height, width, scale), columns, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed image processing coordinator")
    parser.add_argument("--input")
    parser.add_argument("--operation", choices=("upscale", "nlm"), help="image operation to benchmark")
    parser.add_argument("--scale", type=scale_arg)
    parser.add_argument("--nlm-search-window", type=int, help="odd NLM search window; larger values take longer")
    parser.add_argument("--mode", choices=("sequential", "local-parallel", "distributed"))
    parser.add_argument("--worker", action="append", default=[], metavar="HOST[:PORT]")
    parser.add_argument("--workers", type=int, help="limit discovered workers")
    parser.add_argument("--scheduler", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--fault-tolerance", action=argparse.BooleanOptionalAction, default=None, help="retry failed tasks on another Worker")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="core width and height of each square image tile")
    parser.add_argument("--output", default="after")
    args = parser.parse_args()
    if args.tile_size < 1:
        parser.error("--tile-size must be at least 1")
    if args.nlm_search_window is not None and (args.nlm_search_window < 7 or args.nlm_search_window % 2 == 0):
        parser.error("--nlm-search-window must be an odd integer of at least 7")
    workers: list[dict] = []
    planned_worker_count = 0
    # No-argument launch is the guided terminal interface used for demos.
    if not args.input:
        print("=== Distributed Image Processor ===")
        while True:
            entered_path = input("Image path or filename in before/: ").strip()
            candidate = Path(entered_path).expanduser()
            if candidate.is_file():
                args.input = str(candidate)
                break
            before_candidate = Path("before") / candidate.name
            if candidate.parent == Path(".") and before_candidate.is_file():
                args.input = str(before_candidate)
                break
            print("File not found. Enter a valid path or a filename located in before/.")
        if args.operation is None:
            print("\nChoose image operation:\n  1) Upscale: enlarge and enhance the image\n  2) NLM: denoise while keeping the original dimensions")
            operations = {"1": "upscale", "2": "nlm"}
            while args.operation not in operations.values():
                args.operation = operations.get(input("Operation [1-2]: ").strip())
        if args.operation == "upscale":
            while args.scale is None:
                try: args.scale = scale_arg(input("Upscale factor (1.0-10.0): ").strip())
                except (ValueError, argparse.ArgumentTypeError) as exc: print(f"Invalid scale: {exc}")
        else:
            args.scale = 1.0
            if args.nlm_search_window is None:
                while True:
                    entered = input("NLM search window [21, odd number at least 7; larger is slower]: ").strip()
                    if not entered:
                        args.nlm_search_window = 21
                        break
                    try:
                        candidate_window = int(entered)
                        if candidate_window >= 7 and candidate_window % 2 == 1:
                            args.nlm_search_window = candidate_window
                            break
                    except ValueError:
                        pass
                    print("Enter an odd integer of at least 7.")
        print("\nChoose execution mode:\n  1) Sequential baseline: one process on Master\n  2) Local parallel: multiple processes on Master\n  3) Distributed: selected Workers process image tiles concurrently")
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
            print("Scheduler:\n  1) Dynamic: give the next task to whichever Worker slot finishes first\n  2) Static: preassign tasks to process slots; more slots means more assigned tasks")
            args.scheduler = "static" if input("Scheduler [1-2, default 1]: ").strip() == "2" else "dynamic"
            args.fault_tolerance = input("Enable fault tolerance? [y/N]: ").strip().lower() in ("y", "yes")
            print("[FAULT TOLERANCE] Y: failed tasks retry on another Worker, with waits for busy slots. N: no retry; processing continues and missing regions are saved in magenta.")
        print()
    else:
        args.operation = args.operation or "upscale"
        if args.operation == "nlm":
            if args.scale is not None and args.scale != 1.0:
                parser.error("--scale must be 1.0 or omitted when --operation nlm is selected")
            args.scale = 1.0
            args.nlm_search_window = args.nlm_search_window or 21
        elif args.scale is None:
            parser.error("--scale is required for --operation upscale")
    if args.operation == "upscale":
        args.nlm_search_window = args.nlm_search_window or 21
    tile_overlap = max(
        DEFAULT_OVERLAP,
        args.nlm_search_window // 2 + 7 // 2 if args.operation == "nlm" else DEFAULT_OVERLAP,
    )
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
        if args.operation not in {"upscale", "nlm"}:
            parser.error(f"unsupported image operation: {args.operation}")
        unsupported = [w for w in workers if args.operation not in w.get("operations", ["upscale"])]
        if unsupported:
            names = ", ".join(f"{w['host']}@{w['ip']}" for w in unsupported)
            parser.error(f"Workers {names} do not advertise operation '{args.operation}'; update worker.py and restart them")
        print("[MASTER] Selected Workers: " + ", ".join(f"{w['host']}@{w['ip']}" for w in workers))
    image = read_image(args.input)
    h, w, _ = image.shape
    operation_description = (
        f"NLM search_window={args.nlm_search_window}"
        if args.operation == "nlm" else f"upscale scale={args.scale}"
    )
    print(f"[MASTER] {args.input} {w}x{h} operation={operation_description} mode={args.mode}")
    baseline_seconds = None
    if args.mode != "sequential":
        print("[BENCHMARK] Measuring single-process baseline using the same image tiles...")
        baseline_started = time.perf_counter()
        sequential_result, baseline_columns, baseline_rows = process_tiles_sequential(
            image, args.scale, args.tile_size, tile_overlap,
            args.operation, args.nlm_search_window,
        )
        baseline_seconds = time.perf_counter() - baseline_started
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
    task_preparation_seconds = 0.0
    parallel_wall_seconds = 0.0
    stitching_seconds = 0.0
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
        for task in tasks:
            processed = process_slice(task["image"], args.scale, args.operation, args.nlm_search_window)
            pieces[(task["y"], task["x"])] = crop_tile_core(processed, task, args.scale)
        result = stitch_tile_pieces(pieces, h, w, args.scale)
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
            with concurrent.futures.ProcessPoolExecutor() as pool:
                futures = {
                    pool.submit(
                        local_piece,
                        (t["image"].tobytes(), t["image"].shape, args.scale, args.operation, args.nlm_search_window),
                    ): t for t in tasks
                }
                for f, t in futures.items():
                    pieces[(t["y"], t["x"])] = crop_tile_core(f.result(), t, args.scale)
        else:
            print("[MASTER] workers=" + ", ".join(f"{w['host']}@{w['ip']}({w.get('cores', 1)} processes)" for w in workers))
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
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
                active = {}
                def submit(worker, task):
                    task.setdefault("assigned_worker", worker["host"])
                    task.setdefault("assigned_worker_ip", worker["ip"])
                    active[pool.submit(
                        remote_task, worker, task, args.scale, args.operation, args.nlm_search_window
                    )] = (worker, task)
                if args.scheduler == "static":
                    assignments = [[] for _ in slots]
                    for index, task in enumerate(pending_tasks): assignments[index % len(slots)].append(task)
                    pending_tasks.clear()
                    for worker, group in zip(slots, assignments):
                        for task in group: submit(worker, task)
                else:
                    for worker in slots:
                        if pending_tasks: submit(worker, pending_tasks.pop(0))
                while active:
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        worker, task = active.pop(future)
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
                                                candidate, task, args.scale, args.operation, args.nlm_search_window
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
                            task_network = interval_union_seconds(
                                subtract_intervals(task_network_intervals, task_compute_intervals)
                            )
                            task_total = float(transport.get("total", 0.0))
                            task_other = max(0.0, task_total - task_network - task_compute)
                            stats = worker_stats.setdefault(worker["ip"], empty_worker_stats(worker["host"]))
                            stats["tasks"] += 1
                            stats["compute"] += float(response.get("compute_seconds", 0.0))
                            stats["worker_elapsed"] += float(response.get("worker_elapsed", 0.0))
                            phase_intervals = response.get("phase_intervals", {})
                            stats["master_upload_intervals"].extend(clip_intervals(
                                [tuple(transport.get("upload_interval", (0, 0)))], task_window
                            ))
                            stats["master_download_intervals"].extend(clip_intervals(
                                [tuple(transport.get("download_interval", (0, 0)))], task_window
                            ))
                            stats["master_compute_intervals"].extend(task_compute_intervals)
                            stats["master_active_intervals"].extend(task_window)
                            stats["active_intervals"].append(tuple(phase_intervals.get("worker_interval", (0, 0))))
                            stats["task_total"] += float(transport.get("total", 0.0))
                            stats["other"] += task_other
                            for key in ("connect", "upload", "busy_wait", "worker_wait", "download"):
                                stats[key] += float(transport.get(key, 0.0))
                            stats["network_bytes_in"] += int(transport.get("sent_bytes", task["image"].nbytes))
                            stats["network_bytes_out"] += int(transport.get("received_bytes", 0))
                            stats["cpu"].append(float(response.get("cpu_percent", 0.0)))
                            stats["memory"].append(float(response.get("memory_percent", 0.0)))
                            stats["bytes"] += int(task["image"].nbytes)
                            stats["retries"] += int(retried)
                            if retried and worker["ip"] != task.get("assigned_worker_ip"):
                                task_status = f"REROUTED TO {worker['host']}"
                            elif retried:
                                task_status = f"RETRIED ON {worker['host']}"
                            else:
                                task_status = "SUCCESS"
                            task["status"] = task_status
                            region = f"x={task['x']}..{task['x_end']-1},y={task['y']}..{task['y_end']-1}"
                            detail = {"id": task["id"], "label": task["label"], "worker": task.get("assigned_worker", worker["host"]), "status": task_status, "region": region, "bytes_in": transport.get("sent_bytes", task["image"].nbytes), "bytes_out": transport.get("received_bytes", 0), "connect": transport.get("connect", 0), "upload": transport.get("upload", 0), "busy_wait": transport.get("busy_wait", 0), "worker_wait": transport.get("worker_wait", 0), "compute": response.get("compute_seconds", 0), "download": transport.get("download", 0), "total": transport.get("total", 0), "retried": retried}
                            detail["network"] = task_network
                            detail["network_intervals"] = task_network_intervals
                            detail["compute"] = task_compute
                            detail["other"] = task_other
                            task_details.append(detail)
                            print(f"[TASK] {task['label']} -> {worker['host']} tile={region} connect={detail['connect']:.3f}s network={detail['network']:.3f}s busy_wait={detail['busy_wait']:.3f}s worker_wait={detail['worker_wait']:.3f}s compute={detail['compute']:.3f}s task_data={detail['bytes_in']/1e6:.2f}MB result_data={detail['bytes_out']/1e6:.2f}MB")
                            pieces[(task["y"], task["x"])] = crop_tile_core(part, task, args.scale)
                            completed += 1
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
        stitching_started = time.perf_counter()
        if failed_tasks:
            result = np.full((round(h * args.scale), round(w * args.scale), 3), (255, 0, 255), dtype=np.uint8)
            for (task_y, task_x), part in pieces.items():
                out_y = round(task_y * args.scale)
                out_x = round(task_x * args.scale)
                result[out_y:out_y + part.shape[0], out_x:out_x + part.shape[1]] = part
            print(f"[PARTIAL IMAGE] {len(failed_tasks)} task region(s) highlighted magenta.")
        else:
            result = stitch_tile_pieces(pieces, h, w, args.scale)
        stitching_seconds = time.perf_counter() - stitching_started
    elapsed = time.perf_counter() - started
    output_suffix = f"nlm_sw{args.nlm_search_window}" if args.operation == "nlm" else f"{args.scale:.1f}x"
    path = Path(args.output) / f"{Path(args.input).stem}_{output_suffix}.png"
    save_image(path, result)
    status = "PARTIAL" if failed_tasks else "COMPLETE"
    print(f"[MASTER] status={status} operation={args.operation} output={path} shape={result.shape} checksum={checksum(result)} elapsed={elapsed:.3f}s")
    throughput = h * w / elapsed if elapsed else 0.0
    operation_report = (
        f"NLM denoising (search window={args.nlm_search_window}, h=3, hColor=3, template=7)"
        if args.operation == "nlm" else f"Upscale (factor={args.scale}x)"
    )
    report = ["Distributed Image Processing - Benchmark Report", f"Master: {socket.gethostname()}", f"Input: {args.input}", f"Resolution: {w}x{h}", f"Operation: {operation_report}", f"Mode: {args.mode}", f"Scheduler: {args.scheduler}", f"Status: {status}", f"Output: {path}", f"Execution time: {elapsed:.6f}s", f"Throughput: {throughput:.3f} input-pixels/s"]
    parallelism = planned_worker_count if args.mode == "distributed" else (multiprocessing.cpu_count() or 1)
    speedup = baseline_seconds / elapsed if baseline_seconds is not None and elapsed else None
    efficiency = speedup / max(1, parallelism) * 100 if speedup is not None else None
    analysis_rows = [("Operation", operation_report), ("Execution mode", args.mode), ("Scheduler", args.scheduler), ("Worker nodes", str(planned_worker_count)), ("T_seq", f"{baseline_seconds:.3f}s" if baseline_seconds is not None else "baseline run"), ("T_run", f"{elapsed:.3f}s"), ("Speedup", f"{speedup:.3f}x" if speedup is not None else "n/a"), ("Efficiency", f"{efficiency:.2f}% (P={parallelism})" if efficiency is not None else "n/a"), ("Throughput", f"{throughput:.2f} input-pixels/s")]
    print("\n[OVERALL ANALYSIS]")
    print(f"{'METRIC':<24} VALUE")
    report.extend(["", "OVERALL ANALYSIS", f"{'METRIC':<24} VALUE"])
    for label, value in analysis_rows:
        line = f"{label:<24} {value}"
        print(line); report.append(line)
    if args.mode != "sequential":
        all_network_intervals = [interval for stat in worker_stats.values() for interval in (stat["master_upload_intervals"] + stat["master_download_intervals"])]
        all_compute_intervals = [interval for stat in worker_stats.values() for interval in stat["master_compute_intervals"]]
        run_window = [(parallel_started, parallel_started + parallel_wall_seconds)]
        run_network = clip_intervals(all_network_intervals, run_window)
        run_compute = clip_intervals(all_compute_intervals, run_window)
        all_compute = interval_union_seconds(run_compute)
        # Count network/compute overlap as compute time, not network time.
        run_network = subtract_intervals(run_network, run_compute)
        all_network = interval_union_seconds(run_network)
        # Other is the unmeasured remainder, including preparation/stitching.
        run_other = max(0.0, elapsed - all_network - all_compute)
        phase_rows = [
            ("Network (all Workers)", all_network),
            ("Pure compute (all Workers)", all_compute),
            ("Other (whole job)", run_other),
            ("T_run total", elapsed),
        ]
        print("\n[T_RUN PHASE WALL TIMES - NETWORK EXCLUDES COMPUTE OVERLAP]")
        phase_header = f"{'PHASE':<30} {'WALL TIME':>12}"
        print(phase_header)
        report.extend(["", "T_RUN PHASE WALL TIMES - NETWORK EXCLUDES COMPUTE OVERLAP", phase_header])
        for phase, seconds in phase_rows:
            line = f"{phase:<30} {seconds:>11.3f}s"
            print(line); report.append(line)
    if worker_stats:
        header = f"{'WORKER':<22} {'TASKS':>5} {'FAILED':>6} {'NETWORK':>10} {'COMPUTE':>9} {'OTHER (WORKER)':>14} {'ACTIVE':>9} {'AVG CPU':>9} {'AVG RAM':>9} {'TASK MB':>9} {'RESULT MB':>10} {'RETRY':>6}"
        print("\n[WORKER PHASE WALL TIMES - COMPUTE OVERLAP REMOVED]")
        print(header)
        report.extend(["", "WORKER PERFORMANCE", header])
        for stat in worker_stats.values():
            avg_cpu = sum(stat["cpu"]) / len(stat["cpu"]) if stat["cpu"] else 0.0
            avg_mem = sum(stat["memory"]) / len(stat["memory"]) if stat["memory"] else 0.0
            worker_wall = interval_union_seconds(stat["master_active_intervals"])
            active = stat["master_active_intervals"]
            worker_network = clip_intervals(stat["master_upload_intervals"] + stat["master_download_intervals"], active)
            worker_compute = clip_intervals(stat["master_compute_intervals"], active)
            # Use the same job-wide compute overlap removed from T_RUN.
            worker_network = subtract_intervals(worker_network, run_compute)
            network_wall = interval_union_seconds(worker_network)
            compute_wall = interval_union_seconds(worker_compute)
            worker_other = max(0.0, worker_wall - network_wall - compute_wall)
            row = f"{stat['host'][:22]:<22} {stat['tasks']:>5} {stat['failed']:>6} {network_wall:>9.3f}s {compute_wall:>8.3f}s {worker_other:>13.3f}s {worker_wall:>8.3f}s {avg_cpu:>8.1f}% {avg_mem:>8.1f}% {stat['network_bytes_in']/1e6:>8.2f} {stat['network_bytes_out']/1e6:>9.2f} {stat['retries']:>6}"
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
            "bytes_in": task["image"].nbytes,
            "bytes_out": None,
            "network": None,
            "compute": None,
            "other": None,
            "total": None,
        })
    task_header = f"{'TASK':<12} {'WORKER':<22} {'REGION':<27} {'STATUS':<30} {'TASK MB':>8} {'RESULT MB':>10} {'NETWORK':>10} {'NETWORK MB/S':>13} {'COMPUTE':>9} {'OTHER (TASK)':>12} {'TOTAL TIME':>11}"
    print("\n[TASK TIMINGS - NETWORK EXCLUDES JOB-WIDE COMPUTE OVERLAP]"); print(task_header)
    report.extend(["", "TASK TIMINGS - NETWORK EXCLUDES JOB-WIDE COMPUTE OVERLAP", task_header])
    for item in task_details:
        input_mb = item["bytes_in"] / 1e6
        if item["bytes_out"] is None:
            result_mb = network_rate = network = compute = other = total = "-"
        else:
            # Keep task Network on the same basis as the global T_RUN row:
            # any concurrent Worker compute interval is excluded.
            item["network"] = interval_union_seconds(
                subtract_intervals(item["network_intervals"], run_compute)
            )
            item["other"] = max(0.0, item["total"] - item["network"] - item["compute"])
            result_mb = f"{item['bytes_out'] / 1e6:.2f}"
            network_rate = f"{(input_mb + item['bytes_out'] / 1e6) / item['network']:.2f}" if item["network"] else "0.00"
            network = f"{item['network']:.3f}s"
            compute = f"{item['compute']:.3f}s"
            other = f"{item['other']:.3f}s"
            total = f"{item['total']:.3f}s"
        row = f"{item['label']:<12} {item['worker'][:22]:<22} {item['region']:<27} {item['status']:<30} {input_mb:>7.2f} {result_mb:>9} {network:>10} {network_rate:>13} {compute:>9} {other:>12} {total:>11}"
        print(row); report.append(row)
    recovery_line = f"[RECOVERY] retries={sum(x['retries'] for x in worker_stats.values())} recovery_time={recovery_seconds:.3f}s"
    print("\n" + recovery_line); report.append(recovery_line)
    report.extend(failure_events)
    log_dir = Path("logs"); log_dir.mkdir(parents=True, exist_ok=True)
    safe_master = "".join(c if c.isalnum() or c in "-_" else "_" for c in socket.gethostname())
    log_name = f"{safe_master}_{w}x{h}_{output_suffix}_{args.mode}_{args.scheduler}_{len(workers) if args.mode == 'distributed' else 0}workers_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    log_path = log_dir / log_name
    log_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[LOG] benchmark report saved: {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
