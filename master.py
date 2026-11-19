from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import socket
import time
import uuid
from pathlib import Path

import numpy as np

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
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        for target in ("255.255.255.255", "<broadcast>"):
            try: sock.sendto(b"DISCOVER_UPSCAYL_WORKERS_V1", (target, DISCOVERY_PORT))
            except OSError: pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw, addr = sock.recvfrom(2048)
                info = __import__("json").loads(raw.decode())
                if info.get("protocol") == PROTOCOL_VERSION:
                    found[addr[0]] = info | {"ip": addr[0]}
            except socket.timeout: pass
    return list(found.values())


def remote_task(worker: dict, task: dict, scale: float) -> tuple[dict, np.ndarray]:
    image = task["image"]
    with socket.create_connection((worker["ip"], worker["port"]), timeout=SOCKET_TIMEOUT) as sock:
        sock.settimeout(SOCKET_TIMEOUT)
        send_json(sock, {"version": PROTOCOL_VERSION, "task_id": task["id"], "shape": list(image.shape), "scale": scale, "checksum": checksum(image)})
        send_bytes(sock, np.ascontiguousarray(image).tobytes())
        response = recv_json(sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "worker failed"))
        raw = recv_bytes(sock)
    result = np.frombuffer(raw, dtype=np.uint8).reshape(response["shape"]).copy()
    if checksum(result) != response["checksum"]:
        raise ValueError("result checksum mismatch")
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
    parser.add_argument("--rows-per-task", type=int, default=256)
    parser.add_argument("--fail-after", type=int, default=0, help="simulate losing a worker after this many completed tasks")
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
            if args.scheduler == "dynamic" and len(found) > 1 and input("Simulate scheduler removing a Worker after a completed task? (not a real network failure) [y/N]: ").strip().lower() == "y":
                try: args.fail_after = max(1, int(input("Lose a Worker after how many tasks complete? ").strip()))
                except ValueError: args.fail_after = 1
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
    recovery_seconds = 0.0
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
                        try:
                            response, part = future.result()
                        except Exception as first_error:
                            recovery_started = time.perf_counter()
                            recovered = False
                            for candidate in workers:
                                if candidate["ip"] == worker["ip"]: continue
                                try:
                                    response, part = remote_task(candidate, task, args.scale); worker = candidate; recovered = True; retried = True; break
                                except Exception: continue
                            if not recovered: raise RuntimeError(f"task {task['id']} failed; retry unavailable: {first_error}")
                            recovery_seconds += time.perf_counter() - recovery_started
                        stats = worker_stats.setdefault(worker["ip"], {"host": worker["host"], "tasks": 0, "compute": 0.0, "cpu": [], "memory": [], "bytes": 0, "retries": 0})
                        stats["tasks"] += 1
                        stats["compute"] += float(response.get("compute_seconds", 0.0))
                        stats["cpu"].append(float(response.get("cpu_percent", 0.0)))
                        stats["memory"].append(float(response.get("memory_percent", 0.0)))
                        stats["bytes"] += int(task["image"].nbytes)
                        stats["retries"] += int(retried)
                        print(f"[TASK] {task['id'][:8]} -> {worker['host']} compute={response.get('compute_seconds', 0):.2f}s cpu={response.get('cpu_percent', 0):.1f}% mem={response.get('memory_percent', 0):.1f}% retried={retried}")
                        top = round((task["y"] - task["lo"]) * args.scale); bottom = round((task["end"] - task["lo"]) * args.scale)
                        pieces[task["y"]] = part[top:bottom]
                        completed += 1
                        if args.fail_after and completed == args.fail_after:
                            workers = [w for w in workers if w["ip"] != worker["ip"]]
                            slots = [w for w in slots if w["ip"] != worker["ip"]]
                            print(f"[FAULT] simulated loss of {worker['host']} after task {completed}")
                        if args.scheduler == "dynamic" and pending_tasks and slots:
                            submit(worker if worker in slots else slots[0], pending_tasks.pop(0))
                if pending_tasks: raise RuntimeError("no active Worker slots for pending tasks")
        result = np.vstack([pieces[y] for y in sorted(pieces)])
    elapsed = time.perf_counter() - started
    path = Path(args.output) / f"{Path(args.input).stem}_{args.scale:.1f}x.png"
    save_image(path, result)
    print(f"[MASTER] output={path} shape={result.shape} checksum={checksum(result)} elapsed={elapsed:.3f}s")
    print(f"[METRICS] mode={args.mode} scheduler={args.scheduler} workers={len(workers) if args.mode == 'distributed' else 0} seconds={elapsed:.6f} throughput={h * w / elapsed:.2f} input_pixels/s")
    if baseline_seconds is not None:
        parallelism = len(workers) if args.mode == "distributed" else (multiprocessing.cpu_count() or 1)
        speedup = baseline_seconds / elapsed if elapsed else 0.0
        print(f"[ANALYSIS] T_seq={baseline_seconds:.3f}s T_run={elapsed:.3f}s speedup={speedup:.3f}x efficiency={speedup / max(1, parallelism) * 100:.2f}% P={parallelism}")
    if worker_stats:
        print("[WORKER ANALYSIS] host | tasks | compute total | avg CPU | avg memory | input data | retries")
        for stat in worker_stats.values():
            avg_cpu = sum(stat["cpu"]) / len(stat["cpu"]) if stat["cpu"] else 0.0
            avg_mem = sum(stat["memory"]) / len(stat["memory"]) if stat["memory"] else 0.0
            print(f"  {stat['host']} | {stat['tasks']} | {stat['compute']:.3f}s | {avg_cpu:.1f}% | {avg_mem:.1f}% | {stat['bytes']/1e6:.2f}MB | {stat['retries']}")
    print(f"[RECOVERY] retries={sum(x['retries'] for x in worker_stats.values())} recovery_time={recovery_seconds:.3f}s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
