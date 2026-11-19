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


def scale_arg(value: str) -> float:
    scale = float(value)
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise argparse.ArgumentTypeError("scale must be between 1 and 10")
    return scale


def worker_info(host: str, port: int) -> dict:
    with socket.create_connection((host, port), timeout=3) as sock:
        send_json(sock, {"action": "info"})
        return recv_json(sock) | {"ip": host, "port": port}


def discover_workers(timeout: float = 1.5) -> list[dict]:
    found: dict[str, dict] = {}
    interfaces: list[tuple[str, str]] = []
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET or not address.address or not address.netmask:
                continue
            try:
                ip = ipaddress.IPv4Address(address.address)
                if ip.is_loopback or ip.is_link_local:
                    continue
                network = ipaddress.IPv4Network(f"{address.address}/{address.netmask}", strict=False)
                interfaces.append((address.address, str(network.broadcast_address)))
            except ValueError:
                continue
    targets = {(None, "255.255.255.255"), *interfaces}
    for local_ip, broadcast_ip in targets:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(0.2)
                if local_ip:
                    sock.bind((local_ip, 0))
                sock.sendto(b"DISCOVER_UPSCAYL_WORKERS_V1", (broadcast_ip, DISCOVERY_PORT))
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    try:
                        raw, addr = sock.recvfrom(2048)
                        info = json.loads(raw.decode())
                        if info.get("protocol") == PROTOCOL_VERSION:
                            found[addr[0]] = info | {"ip": addr[0]}
                    except socket.timeout:
                        continue
        except OSError as exc:
            print(f"[DISCOVERY] broadcast {broadcast_ip} via {local_ip or 'default route'} failed: {exc}")
    return list(found.values())


def remote_task(worker: dict, task: dict, scale: float) -> tuple[dict, np.ndarray]:
    image = task["image"]
    task_started = time.perf_counter()
    deadline = time.monotonic() + SOCKET_TIMEOUT
    connect_seconds = upload_seconds = busy_wait_seconds = 0.0
    payload = np.ascontiguousarray(image).tobytes()
    while True:
        connect_started = time.perf_counter()
        with socket.create_connection((worker["ip"], worker["port"]), timeout=SOCKET_TIMEOUT) as sock:
            connect_seconds += time.perf_counter() - connect_started
            sock.settimeout(SOCKET_TIMEOUT)
            send_json(sock, {"version": PROTOCOL_VERSION, "task_id": task["id"], "shape": list(image.shape), "scale": scale, "checksum": checksum(image)})
            acknowledgement = recv_json(sock)
            if acknowledgement.get("busy"):
                delay = min(float(acknowledgement.get("retry_after", 0.5)), max(0.0, deadline - time.monotonic()))
                busy_wait_seconds += delay
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"worker {worker['host']} remained busy past task timeout")
                print(f"[WORKER BUSY] {worker['host']} has no free slot; retrying task {task['id'][:8]} in {delay:.1f}s")
                time.sleep(delay)
                continue
            if not acknowledgement.get("accepted"):
                raise RuntimeError(acknowledgement.get("error", "worker rejected task"))
            upload_started = time.perf_counter()
            send_bytes(sock, payload)
            upload_seconds += time.perf_counter() - upload_started
            response_started = time.perf_counter()
            response = recv_json(sock)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "worker failed"))
            response_wait_seconds = time.perf_counter() - response_started
            download_started = time.perf_counter()
            raw = recv_bytes(sock)
            download_seconds = time.perf_counter() - download_started
        break
    result = np.frombuffer(raw, dtype=np.uint8).reshape(response["shape"]).copy()
    if checksum(result) != response["checksum"]:
        raise ValueError("result checksum mismatch")
    response["transport"] = {"connect": connect_seconds, "upload": upload_seconds, "busy_wait": busy_wait_seconds, "worker_wait": response_wait_seconds, "download": download_seconds, "total": time.perf_counter() - task_started, "sent_bytes": len(payload), "received_bytes": len(raw)}
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
    parser.add_argument("--rows-per-task", type=int, default=256)
    parser.add_argument("--output", default="after")
    args = parser.parse_args()
    # No-argument launch is the guided terminal interface used for demos.
    if not args.input:
        print("=== Distributed Image Upscaler ===\n[MASTER] Scanning LAN for Workers...")
        found = discover_workers()
        for i, worker in enumerate(found, 1):
            print(f"  [{i}] {worker['host']}  {worker['ip']}:{worker['port']}  processes={worker.get('slots', worker.get('cores', 1))}")
        if not found: print("  No Workers discovered. Start worker.py first; you can still choose local modes.")
        while True:
            args.input = input("Image path: ").strip()
            if Path(args.input).is_file(): break
            print("File not found; enter a valid image path.")
        while args.scale is None:
            try: args.scale = scale_arg(input("Upscale factor (1.0-10.0): ").strip())
            except (ValueError, argparse.ArgumentTypeError) as exc: print(f"Invalid scale: {exc}")
        print("\nChoose execution mode:\n  1) Sequential baseline: one process on Master\n  2) Local parallel: multiple processes on Master\n  3) Distributed: selected Workers process strips concurrently")
        modes = {"1": "sequential", "2": "local-parallel", "3": "distributed"}
        while args.mode not in modes.values(): args.mode = modes.get(input("Mode [1-3]: ").strip())
        if args.mode == "distributed":
            if not found:
                print("No discovered nodes; enter Worker addresses as host:port separated by commas (blank to cancel).")
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
            print("Scheduler:\n  1) Dynamic: give a free Worker slot another task immediately\n  2) Static: assign all strips to Worker slots before processing")
            args.scheduler = "static" if input("Scheduler [1-2, default 1]: ").strip() == "2" else "dynamic"
            args.fault_tolerance = input("Enable fault tolerance? [y/N]: ").strip().lower() in ("y", "yes")
            print("[FAULT TOLERANCE] Enabled: failed tasks retry on another Worker. Disabled: a failed task stops the job; no partial output is saved.")
        print()
    args.mode = args.mode or "distributed"
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
    workers: list[dict] = []
    recovery_seconds = 0.0
    failed_tasks: list[dict] = []
    started = time.perf_counter()
    if args.mode == "sequential":
        result = process_slice(image, args.scale)
    else:
        tasks = []
        size = max(1, args.rows_per_task)
        for y in range(0, h, size):
            # Context rows reduce boundary artifacts; crop overlap after scaling.
            lo, hi = max(0, y - 16), min(h, y + size + 16)
            tasks.append({"id": uuid.uuid4().hex, "y": y, "end": min(h, y + size), "lo": lo, "image": image[lo:hi]})
        pieces: dict[int, np.ndarray] = {}
        if args.mode == "local-parallel":
            with concurrent.futures.ProcessPoolExecutor() as pool:
                futures = {pool.submit(local_piece, (t["image"].tobytes(), t["image"].shape, args.scale)): t for t in tasks}
                for f, t in futures.items():
                    part = f.result(); top = round((t["y"] - t["lo"]) * args.scale); bottom = round((t["end"] - t["lo"]) * args.scale)
                    pieces[t["y"]] = part[top:bottom]
        else:
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
                parser.error("no workers discovered; start worker.py on LAN or pass --worker HOST[:PORT]")
            print("[MASTER] workers=" + ", ".join(f"{w['host']}@{w['ip']}({w.get('cores', 1)} processes)" for w in workers))
            pending_tasks = list(tasks)
            slots = [w for w in workers for _ in range(max(1, int(w.get("slots", w.get("cores", 1))))) ]
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
                        try:
                            response, part = future.result()
                        except Exception as first_error:
                            failed_worker = worker
                            failure_line = f"[TASK FAILED] task={task['id']} worker={failed_worker['host']}@{failed_worker['ip']} reason={type(first_error).__name__}: {first_error}"
                            print(failure_line); failure_events.append(failure_line)
                            workers = [w for w in workers if w["ip"] != failed_worker["ip"]]
                            slots = [w for w in slots if w["ip"] != failed_worker["ip"]]
                            if not args.fault_tolerance:
                                failed_tasks.append(task); lost = True
                                print(f"[PARTIAL] task={task['id']} marked missing; its output region will be highlighted")
                            else:
                                recovery_started = time.perf_counter()
                                recovered = False
                                for candidate in workers:
                                    if candidate["ip"] == failed_worker["ip"]: continue
                                    retry_line = f"[RETRY] task={task['id']} rerouted {failed_worker['host']} -> {candidate['host']}@{candidate['ip']}"
                                    print(retry_line); failure_events.append(retry_line)
                                    try:
                                        response, part = remote_task(candidate, task, args.scale); worker = candidate; recovered = True; retried = True; break
                                    except Exception as retry_error:
                                        retry_error_line = f"[RETRY FAILED] task={task['id']} worker={candidate['host']} reason={type(retry_error).__name__}: {retry_error}"
                                        print(retry_error_line); failure_events.append(retry_error_line)
                                if not recovered:
                                    failed_tasks.append(task); lost = True
                                    print(f"[PARTIAL] task={task['id']} has no available retry destination; output region will be highlighted")
                                else:
                                    task_recovery = time.perf_counter() - recovery_started
                                    recovery_seconds += task_recovery
                                    print(f"[RECOVERED] task={task['id']} completed_on={worker['host']} recovery={task_recovery:.3f}s")
                        if not lost:
                            transport = response.get("transport", {})
                            stats = worker_stats.setdefault(worker["ip"], {"host": worker["host"], "tasks": 0, "compute": 0.0, "worker_elapsed": 0.0, "cpu": [], "memory": [], "bytes": 0, "retries": 0, "connect": 0.0, "upload": 0.0, "busy_wait": 0.0, "worker_wait": 0.0, "download": 0.0, "network_bytes_in": 0, "network_bytes_out": 0})
                            stats["tasks"] += 1
                            stats["compute"] += float(response.get("compute_seconds", 0.0))
                            stats["worker_elapsed"] += float(response.get("worker_elapsed", 0.0))
                            for key in ("connect", "upload", "busy_wait", "worker_wait", "download"):
                                stats[key] += float(transport.get(key, 0.0))
                            stats["network_bytes_in"] += int(transport.get("sent_bytes", task["image"].nbytes))
                            stats["network_bytes_out"] += int(transport.get("received_bytes", 0))
                            stats["cpu"].append(float(response.get("cpu_percent", 0.0)))
                            stats["memory"].append(float(response.get("memory_percent", 0.0)))
                            stats["bytes"] += int(task["image"].nbytes)
                            stats["retries"] += int(retried)
                            detail = {"id": task["id"], "worker": worker["host"], "rows": f"{task['y']}..{task['end']-1}", "bytes_in": transport.get("sent_bytes", task["image"].nbytes), "bytes_out": transport.get("received_bytes", 0), "connect": transport.get("connect", 0), "upload": transport.get("upload", 0), "busy_wait": transport.get("busy_wait", 0), "worker_wait": transport.get("worker_wait", 0), "compute": response.get("compute_seconds", 0), "download": transport.get("download", 0), "total": transport.get("total", 0), "retried": retried}
                            task_details.append(detail)
                            print(f"[TASK] {task['id'][:8]} -> {worker['host']} rows={detail['rows']} connect={detail['connect']:.3f}s upload={detail['upload']:.3f}s busy_wait={detail['busy_wait']:.3f}s worker_wait={detail['worker_wait']:.3f}s compute={detail['compute']:.3f}s download={detail['download']:.3f}s in={detail['bytes_in']/1e6:.2f}MB out={detail['bytes_out']/1e6:.2f}MB")
                            top = round((task["y"] - task["lo"]) * args.scale); bottom = round((task["end"] - task["lo"]) * args.scale)
                            pieces[task["y"]] = part[top:bottom]
                            completed += 1
                        if args.scheduler == "dynamic" and pending_tasks and slots:
                            submit(worker if worker in slots else slots[0], pending_tasks.pop(0))
                if pending_tasks: raise RuntimeError("no active Worker slots for pending tasks")
        if failed_tasks:
            output_shape = (round(h * args.scale), round(w * args.scale), 3)
            result = np.full(output_shape, (255, 0, 255), dtype=np.uint8)
            for task_y, part in pieces.items():
                out_y = round(task_y * args.scale)
                result[out_y:out_y + part.shape[0]] = part
            print(f"[PARTIAL IMAGE] {len(failed_tasks)} task region(s) highlighted magenta.")
        else:
            result = np.vstack([pieces[y] for y in sorted(pieces)])
    elapsed = time.perf_counter() - started
    path = Path(args.output) / f"{Path(args.input).stem}_{args.scale:.1f}x.png"
    save_image(path, result)
    status = "PARTIAL" if failed_tasks else "COMPLETE"
    print(f"[MASTER] status={status} output={path} shape={result.shape} checksum={checksum(result)} elapsed={elapsed:.3f}s")
    throughput = h * w / elapsed if elapsed else 0.0
    print(f"[METRICS] mode={args.mode} scheduler={args.scheduler} workers={len(workers) if args.mode == 'distributed' else 0} execution={elapsed:.3f}s throughput={throughput:.2f} input-pixels/s")
    report = ["Distributed Image Upscaler - Benchmark Report", f"Master: {socket.gethostname()}", f"Input: {args.input}", f"Resolution: {w}x{h}", f"Scale: {args.scale}x", f"Mode: {args.mode}", f"Scheduler: {args.scheduler}", f"Status: {status}", f"Output: {path}", f"Execution time: {elapsed:.6f}s", f"Throughput: {throughput:.3f} input-pixels/s"]
    if baseline_seconds is not None:
        parallelism = len(workers) if args.mode == "distributed" else (multiprocessing.cpu_count() or 1)
        speedup = baseline_seconds / elapsed if elapsed else 0.0
        efficiency = speedup / max(1, parallelism) * 100
        analysis = f"[ANALYSIS] T_seq={baseline_seconds:.3f}s T_run={elapsed:.3f}s speedup={speedup:.3f}x efficiency={efficiency:.2f}% P={parallelism}"
        print(analysis)
        report.extend([f"Sequential baseline: {baseline_seconds:.6f}s", f"Speedup: {speedup:.6f}x", f"Efficiency: {efficiency:.3f}% (P={parallelism})"])
    if worker_stats:
        header = f"{'WORKER':<22} {'TASKS':>5} {'COMPUTE':>9} {'UPLOAD':>9} {'BUSY':>8} {'WAIT':>9} {'DOWNLOAD':>9} {'AVG CPU':>9} {'AVG MEM':>9} {'IN MB':>9} {'OUT MB':>9} {'RETRY':>6}"
        print("\n[WORKER ANALYSIS]")
        print(header)
        report.extend(["", "WORKER PERFORMANCE", header])
        for stat in worker_stats.values():
            avg_cpu = sum(stat["cpu"]) / len(stat["cpu"]) if stat["cpu"] else 0.0
            avg_mem = sum(stat["memory"]) / len(stat["memory"]) if stat["memory"] else 0.0
            row = f"{stat['host'][:22]:<22} {stat['tasks']:>5} {stat['compute']:>8.3f}s {stat['upload']:>8.3f}s {stat['busy_wait']:>7.3f}s {stat['worker_wait']:>8.3f}s {stat['download']:>8.3f}s {avg_cpu:>8.1f}% {avg_mem:>8.1f}% {stat['network_bytes_in']/1e6:>8.2f} {stat['network_bytes_out']/1e6:>8.2f} {stat['retries']:>6}"
            print(row); report.append(row)
    task_header = f"{'TASK':<10} {'WORKER':<22} {'ROWS':<15} {'CONNECT':>9} {'UPLOAD':>9} {'BUSY WAIT':>10} {'WORKER WAIT':>12} {'COMPUTE':>9} {'DOWNLOAD':>9} {'TOTAL':>9}"
    print("\n[TASK TIMINGS]"); print(task_header)
    report.extend(["", "TASK TIMINGS", task_header])
    for item in task_details:
        row = f"{item['id'][:8]:<10} {item['worker'][:22]:<22} {item['rows']:<15} {item['connect']:>8.3f}s {item['upload']:>8.3f}s {item['busy_wait']:>9.3f}s {item['worker_wait']:>11.3f}s {item['compute']:>8.3f}s {item['download']:>8.3f}s {item['total']:>8.3f}s"
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
