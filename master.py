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

from config import DEFAULT_PORT, DISCOVERY_PORT, MAX_SCALE, MIN_SCALE, PROTOCOL_VERSION, SOCKET_TIMEOUT
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


def remote_task(worker: dict, task: dict, scale: float, timeout: float = SOCKET_TIMEOUT) -> tuple[dict, np.ndarray]:
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
        send_json(sock, {"version": PROTOCOL_VERSION, "task_id": task["id"], "shape": list(image.shape), "scale": scale, "checksum": checksum(image)})
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


def local_piece(args: tuple[bytes, tuple[int, ...], float]) -> np.ndarray:
    raw, shape, scale = args
    return process_slice(np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(), scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed image upscaler coordinator")
    parser.add_argument("--input")
    parser.add_argument("--scale", type=scale_arg)
    parser.add_argument("--mode", choices=("sequential", "local-parallel", "distributed"))
    parser.add_argument("--worker", action="append", default=[], metavar="HOST[:PORT]")
    parser.add_argument("--workers", type=int, help="limit discovered workers")
    parser.add_argument("--scheduler", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--fault-tolerance", action=argparse.BooleanOptionalAction, default=None, help="retry failed tasks on another Worker")
    parser.add_argument("--rows-per-task", type=int, default=0, help="set manually; 0 selects task count from detected process slots")
    parser.add_argument("--output", default="after")
    args = parser.parse_args()
    workers: list[dict] = []
    planned_worker_count = 0
    # No-argument launch is the guided terminal interface used for demos.
    if not args.input:
        print("=== Distributed Image Upscaler ===")
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
        while args.scale is None:
            try: args.scale = scale_arg(input("Upscale factor (1.0-10.0): ").strip())
            except (ValueError, argparse.ArgumentTypeError) as exc: print(f"Invalid scale: {exc}")
        print("\nChoose execution mode:\n  1) Sequential baseline: one process on Master\n  2) Local parallel: multiple processes on Master\n  3) Distributed: selected Workers process strips concurrently")
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
            print("Scheduler:\n  1) Dynamic: give a free Worker slot another task immediately\n  2) Static: assign all strips to Worker slots before processing")
            args.scheduler = "static" if input("Scheduler [1-2, default 1]: ").strip() == "2" else "dynamic"
            args.fault_tolerance = input("Enable fault tolerance? [y/N]: ").strip().lower() in ("y", "yes")
            print("[FAULT TOLERANCE] Y: failed tasks retry on another Worker, with waits for busy slots. N: no retry; processing continues and missing regions are saved in magenta.")
        print()
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
        print("[MASTER] Selected Workers: " + ", ".join(f"{w['host']}@{w['ip']}" for w in workers))
    image = read_image(args.input)
    h, w, _ = image.shape
    print(f"[MASTER] {args.input} {w}x{h} mode={args.mode} scale={args.scale}")
    baseline_seconds = None
    if args.mode != "sequential":
        print("[BENCHMARK] Measuring sequential baseline...")
        baseline_started = time.perf_counter()
        sequential_result = process_slice(image, args.scale)
        baseline_seconds = time.perf_counter() - baseline_started
        print(f"[BENCHMARK] sequential={baseline_seconds:.3f}s checksum={checksum(sequential_result)}")
    worker_stats: dict[str, dict] = {}
    task_details: list[dict] = []
    failure_events: list[str] = []
    recovery_seconds = 0.0
    failed_tasks: list[dict] = []
    started = time.perf_counter()
    task_preparation_seconds = 0.0
    parallel_wall_seconds = 0.0
    stitching_seconds = 0.0
    if args.mode == "sequential":
        result = process_slice(image, args.scale)
    else:
        preparation_started = time.perf_counter()
        tasks = []
        if args.rows_per_task > 0:
            size = args.rows_per_task
            task_count = (h + size - 1) // size
        else:
            process_slots = (sum(max(1, int(worker.get("slots", worker.get("cores", 1)))) for worker in workers)
                             if args.mode == "distributed" else (multiprocessing.cpu_count() or 1))
            # Create about two tasks per process slot for dynamic balancing,
            # while keeping strips at least roughly 64 source rows tall.
            task_count = min(max(1, process_slots * 2), max(1, (h + 63) // 64))
            size = max(1, (h + task_count - 1) // task_count)
        print(f"[TASK DECOMPOSITION] rows_per_task={size} task_count={(h + size - 1) // size} process_slots={process_slots if args.rows_per_task <= 0 else 'manual'}")
        for y in range(0, h, size):
            # Context rows reduce boundary artifacts; crop overlap after scaling.
            lo, hi = max(0, y - 16), min(h, y + size + 16)
            tasks.append({"id": uuid.uuid4().hex, "y": y, "end": min(h, y + size), "lo": lo, "image": image[lo:hi]})
        task_preparation_seconds = time.perf_counter() - preparation_started
        pieces: dict[int, np.ndarray] = {}
        parallel_started = time.perf_counter()
        if args.mode == "local-parallel":
            with concurrent.futures.ProcessPoolExecutor() as pool:
                futures = {pool.submit(local_piece, (t["image"].tobytes(), t["image"].shape, args.scale)): t for t in tasks}
                for f, t in futures.items():
                    part = f.result(); top = round((t["y"] - t["lo"]) * args.scale); bottom = round((t["end"] - t["lo"]) * args.scale)
                    pieces[t["y"]] = part[top:bottom]
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
                    active[pool.submit(remote_task, worker, task, args.scale)] = (worker, task)
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
                            event = "WORKER BUSY" if isinstance(first_error, WorkerBusyError) else "TASK FAILED"
                            failure_line = f"[{event}] task={task['id']} worker={failed_worker['host']}@{failed_worker['ip']} reason={type(first_error).__name__}: {first_error}"
                            print(failure_line); failure_events.append(failure_line)
                            if not args.fault_tolerance:
                                workers = [w for w in workers if w["ip"] != failed_worker["ip"]]
                                slots = [w for w in slots if w["ip"] != failed_worker["ip"]]
                                failed_tasks.append(task); lost = True
                                print(f"[PARTIAL] task={task['id']} marked missing; its output region will be highlighted")
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
                                        retry_line = f"[RETRY] task={task['id']} attempt={retry_round} target={candidate['host']}@{candidate['ip']}"
                                        print(retry_line); failure_events.append(retry_line)
                                        try:
                                            response, part = remote_task(candidate, task, args.scale)
                                            worker = candidate; recovered = True; retried = True
                                            break
                                        except WorkerBusyError as retry_error:
                                            saw_busy = True
                                            busy_line = f"[RETRY WAIT] task={task['id']} worker={candidate['host']} is busy; will try again"
                                            print(busy_line); failure_events.append(busy_line)
                                        except Exception as retry_error:
                                            retry_error_line = f"[RETRY FAILED] task={task['id']} worker={candidate['host']} reason={type(retry_error).__name__}: {retry_error}"
                                            print(retry_error_line); failure_events.append(retry_error_line)
                                    if recovered:
                                        break
                                    if retry_round < 4:
                                        delay = min(2 ** (retry_round - 1), 8)
                                        wait_line = f"[RETRY BACKOFF] task={task['id']} round={retry_round} wait={delay}s" + (" (Workers busy)" if saw_busy else "")
                                        print(wait_line); failure_events.append(wait_line)
                                        time.sleep(delay)
                                if not recovered:
                                    failed_tasks.append(task); lost = True
                                    print(f"[PARTIAL] task={task['id']} retry limit reached; output region will be highlighted")
                                else:
                                    task_recovery = time.perf_counter() - recovery_started
                                    recovery_seconds += task_recovery
                                    print(f"[RECOVERED] task={task['id']} completed_on={worker['host']} recovery={task_recovery:.3f}s")
                        if not lost:
                            transport = response.get("transport", {})
                            stats = worker_stats.setdefault(worker["ip"], {"host": worker["host"], "tasks": 0, "compute": 0.0, "task_total": 0.0, "other": 0.0, "worker_elapsed": 0.0, "cpu": [], "memory": [], "bytes": 0, "retries": 0, "connect": 0.0, "upload": 0.0, "busy_wait": 0.0, "worker_wait": 0.0, "download": 0.0, "network_bytes_in": 0, "network_bytes_out": 0, "compute_intervals": [], "upload_intervals": [], "download_intervals": [], "active_intervals": [], "master_upload_intervals": [], "master_download_intervals": [], "master_compute_intervals": [], "master_active_intervals": []})
                            stats["tasks"] += 1
                            stats["compute"] += float(response.get("compute_seconds", 0.0))
                            stats["worker_elapsed"] += float(response.get("worker_elapsed", 0.0))
                            phase_intervals = response.get("phase_intervals", {})
                            stats["master_upload_intervals"].append(tuple(transport.get("upload_interval", (0, 0))))
                            stats["master_download_intervals"].append(tuple(transport.get("download_interval", (0, 0))))
                            stats["master_compute_intervals"].append(tuple(transport.get("compute_interval", (0, 0))))
                            stats["master_active_intervals"].append(tuple(transport.get("task_interval", (0, 0))))
                            stats["active_intervals"].append(tuple(phase_intervals.get("worker_interval", (0, 0))))
                            stats["task_total"] += float(transport.get("total", 0.0))
                            stats["other"] += max(0.0, float(transport.get("total", 0.0)) - float(transport.get("upload", 0.0)) - float(response.get("compute_seconds", 0.0)) - float(transport.get("download", 0.0)))
                            for key in ("connect", "upload", "busy_wait", "worker_wait", "download"):
                                stats[key] += float(transport.get(key, 0.0))
                            stats["network_bytes_in"] += int(transport.get("sent_bytes", task["image"].nbytes))
                            stats["network_bytes_out"] += int(transport.get("received_bytes", 0))
                            stats["cpu"].append(float(response.get("cpu_percent", 0.0)))
                            stats["memory"].append(float(response.get("memory_percent", 0.0)))
                            stats["bytes"] += int(task["image"].nbytes)
                            stats["retries"] += int(retried)
                            detail = {"id": task["id"], "worker": worker["host"], "rows": f"{task['y']}..{task['end']-1}", "bytes_in": transport.get("sent_bytes", task["image"].nbytes), "bytes_out": transport.get("received_bytes", 0), "connect": transport.get("connect", 0), "upload": transport.get("upload", 0), "busy_wait": transport.get("busy_wait", 0), "worker_wait": transport.get("worker_wait", 0), "compute": response.get("compute_seconds", 0), "download": transport.get("download", 0), "total": transport.get("total", 0), "retried": retried}
                            detail["network"] = detail["upload"] + detail["download"]
                            detail["other"] = max(0.0, detail["total"] - detail["network"] - detail["compute"])
                            task_details.append(detail)
                            print(f"[TASK] {task['id'][:8]} -> {worker['host']} rows={detail['rows']} connect={detail['connect']:.3f}s network={detail['network']:.3f}s busy_wait={detail['busy_wait']:.3f}s worker_wait={detail['worker_wait']:.3f}s compute={detail['compute']:.3f}s task_data={detail['bytes_in']/1e6:.2f}MB result_data={detail['bytes_out']/1e6:.2f}MB")
                            top = round((task["y"] - task["lo"]) * args.scale); bottom = round((task["end"] - task["lo"]) * args.scale)
                            pieces[task["y"]] = part[top:bottom]
                            completed += 1
                        if args.scheduler == "dynamic" and pending_tasks and slots:
                            submit(worker if worker in slots else slots[0], pending_tasks.pop(0))
                if pending_tasks:
                    if args.fault_tolerance:
                        raise RuntimeError("no active Worker slots for pending tasks")
                    failed_tasks.extend(pending_tasks)
                    print(f"[PARTIAL] {len(pending_tasks)} pending task(s) have no available Worker and will be highlighted")
                    pending_tasks.clear()
        parallel_wall_seconds = time.perf_counter() - parallel_started
        stitching_started = time.perf_counter()
        if failed_tasks:
            output_shape = (round(h * args.scale), round(w * args.scale), 3)
            result = np.full(output_shape, (255, 0, 255), dtype=np.uint8)
            for task_y, part in pieces.items():
                out_y = round(task_y * args.scale)
                result[out_y:out_y + part.shape[0]] = part
            print(f"[PARTIAL IMAGE] {len(failed_tasks)} task region(s) highlighted magenta.")
        else:
            result = np.vstack([pieces[y] for y in sorted(pieces)])
        stitching_seconds = time.perf_counter() - stitching_started
    elapsed = time.perf_counter() - started
    path = Path(args.output) / f"{Path(args.input).stem}_{args.scale:.1f}x.png"
    save_image(path, result)
    status = "PARTIAL" if failed_tasks else "COMPLETE"
    print(f"[MASTER] status={status} output={path} shape={result.shape} checksum={checksum(result)} elapsed={elapsed:.3f}s")
    throughput = h * w / elapsed if elapsed else 0.0
    report = ["Distributed Image Upscaler - Benchmark Report", f"Master: {socket.gethostname()}", f"Input: {args.input}", f"Resolution: {w}x{h}", f"Scale: {args.scale}x", f"Mode: {args.mode}", f"Scheduler: {args.scheduler}", f"Status: {status}", f"Output: {path}", f"Execution time: {elapsed:.6f}s", f"Throughput: {throughput:.3f} input-pixels/s"]
    parallelism = planned_worker_count if args.mode == "distributed" else (multiprocessing.cpu_count() or 1)
    speedup = baseline_seconds / elapsed if baseline_seconds is not None and elapsed else None
    efficiency = speedup / max(1, parallelism) * 100 if speedup is not None else None
    analysis_rows = [("Execution mode", args.mode), ("Scheduler", args.scheduler), ("Worker nodes", str(planned_worker_count)), ("T_seq", f"{baseline_seconds:.3f}s" if baseline_seconds is not None else "baseline run"), ("T_run", f"{elapsed:.3f}s"), ("Speedup", f"{speedup:.3f}x" if speedup is not None else "n/a"), ("Efficiency", f"{efficiency:.2f}% (P={parallelism})" if efficiency is not None else "n/a"), ("Throughput", f"{throughput:.2f} input-pixels/s")]
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
        all_network = interval_union_seconds(run_network)
        all_compute = interval_union_seconds(run_compute)
        # Other is the balancing remainder: it includes setup/stitching and
        # subtracts network/compute overlap so the phase rows add to T_run.
        run_other = elapsed - all_network - all_compute
        phase_rows = [
            ("Network (all Workers)", all_network),
            ("Pure compute (all Workers)", all_compute),
            ("Other", run_other),
            ("T_run total", elapsed),
        ]
        print("\n[T_RUN PHASE WALL TIMES - OTHER INCLUDES OVERLAP ADJUSTMENT]")
        phase_header = f"{'PHASE':<30} {'WALL TIME':>12}"
        print(phase_header)
        report.extend(["", "T_RUN PHASE WALL TIMES - OTHER INCLUDES OVERLAP ADJUSTMENT", phase_header])
        for phase, seconds in phase_rows:
            line = f"{phase:<30} {seconds:>11.3f}s"
            print(line); report.append(line)
    if worker_stats:
        header = f"{'WORKER':<22} {'TASKS':>5} {'NETWORK':>10} {'COMPUTE':>9} {'OTHER':>9} {'ACTIVE':>9} {'AVG CPU':>9} {'AVG RAM':>9} {'TASK MB':>9} {'RESULT MB':>10} {'RETRY':>6}"
        print("\n[WORKER PHASE WALL TIMES - PER WORKER; MAY OVERLAP]")
        print(header)
        report.extend(["", "WORKER PERFORMANCE", header])
        for stat in worker_stats.values():
            avg_cpu = sum(stat["cpu"]) / len(stat["cpu"]) if stat["cpu"] else 0.0
            avg_mem = sum(stat["memory"]) / len(stat["memory"]) if stat["memory"] else 0.0
            worker_wall = interval_union_seconds(stat["master_active_intervals"])
            active = stat["master_active_intervals"]
            worker_network = clip_intervals(stat["master_upload_intervals"] + stat["master_download_intervals"], active)
            worker_compute = clip_intervals(stat["master_compute_intervals"], active)
            network_wall = interval_union_seconds(worker_network)
            compute_wall = interval_union_seconds(worker_compute)
            covered_worker = interval_union_seconds(worker_network + worker_compute)
            worker_other = max(0.0, worker_wall - covered_worker)
            row = f"{stat['host'][:22]:<22} {stat['tasks']:>5} {network_wall:>9.3f}s {compute_wall:>8.3f}s {worker_other:>8.3f}s {worker_wall:>8.3f}s {avg_cpu:>8.1f}% {avg_mem:>8.1f}% {stat['network_bytes_in']/1e6:>8.2f} {stat['network_bytes_out']/1e6:>9.2f} {stat['retries']:>6}"
            print(row); report.append(row)
    task_header = f"{'TASK':<10} {'WORKER':<22} {'ROWS':<15} {'TASK MB':>8} {'RESULT MB':>10} {'NETWORK':>10} {'NETWORK MB/S':>13} {'COMPUTE':>9} {'OTHER':>9} {'TOTAL TIME':>11}"
    print("\n[TASK TIMINGS]"); print(task_header)
    report.extend(["", "TASK TIMINGS", task_header])
    for item in task_details:
        input_mb = item["bytes_in"] / 1e6
        result_mb = item["bytes_out"] / 1e6
        network_rate = (input_mb + result_mb) / item["network"] if item["network"] else 0.0
        row = f"{item['id'][:8]:<10} {item['worker'][:22]:<22} {item['rows']:<15} {input_mb:>7.2f} {result_mb:>9.2f} {item['network']:>9.3f}s {network_rate:>12.2f} {item['compute']:>8.3f}s {item['other']:>8.3f}s {item['total']:>10.3f}s"
        print(row); report.append(row)
    recovery_line = f"[RECOVERY] retries={sum(x['retries'] for x in worker_stats.values())} recovery_time={recovery_seconds:.3f}s"
    print("\n" + recovery_line); report.append(recovery_line)
    report.extend(failure_events)
    log_dir = Path("logs"); log_dir.mkdir(parents=True, exist_ok=True)
    safe_master = "".join(c if c.isalnum() or c in "-_" else "_" for c in socket.gethostname())
    log_name = f"{safe_master}_{w}x{h}_{args.scale:.1f}x_{args.mode}_{args.scheduler}_{len(workers) if args.mode == 'distributed' else 0}workers_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    log_path = log_dir / log_name
    log_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[LOG] benchmark report saved: {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
