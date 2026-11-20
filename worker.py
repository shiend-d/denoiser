from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import threading
import socket
import time

import numpy as np
import psutil

from config import DEFAULT_PORT, DISCOVERY_PORT, PROTOCOL_VERSION, SOCKET_TIMEOUT
from image_pipeline import checksum, process_slice
from protocol import recv_bytes, recv_json, send_bytes, send_json


def _process(
    payload: bytes, shape: tuple[int, ...],
    nlm_search_window: int, nlm_h: float, nlm_h_color: float,
) -> tuple[bytes, tuple[int, ...], str, float]:
    started = time.perf_counter()
    image = np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
    result = process_slice(image, nlm_search_window, nlm_h, nlm_h_color)
    result_bytes = np.ascontiguousarray(result).tobytes()
    return result_bytes, result.shape, checksum(result), time.perf_counter() - started


def handle(conn: socket.socket, address: tuple[str, int], pool: concurrent.futures.ProcessPoolExecutor, capacity: threading.BoundedSemaphore) -> None:
    with conn:
        conn.settimeout(SOCKET_TIMEOUT)
        reserved = False
        task_label = "?/?"
        master_label = f"Master@{address[0]}"
        try:
            request = recv_json(conn)
            if request.get("action") == "info":
                send_json(conn, {"ok": True, "host": socket.gethostname(), "cores": multiprocessing.cpu_count(), "slots": multiprocessing.cpu_count(), "operations": ["nlm"], "nlm_task_schema": 4})
                return
            if request.get("version") != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol version")
            if request.get("operation") != "nlm":
                raise ValueError("unsupported or missing image operation; this Worker supports NLM only")
            if request.get("nlm_task_schema") != 4:
                raise ValueError("incompatible NLM task schema; update the Master")
            task_label = f"{request.get('task_number', '?')}/{request.get('task_count', '?')}"
            master_label = f"{request.get('master_host', 'Master')}@{address[0]}"
            if not capacity.acquire(blocking=False):
                send_json(conn, {"ok": False, "busy": True, "retry_after": 0.5, "error": "all worker process slots are busy"})
                print(f"[BUSY] task={task_label} master={master_label}", flush=True)
                return
            reserved = True
            send_json(conn, {"ok": True, "accepted": True})
            input_receive_started = time.perf_counter()
            payload = recv_bytes(conn)
            input_receive_seconds = time.perf_counter() - input_receive_started
            if checksum(np.frombuffer(payload, dtype=np.uint8)) != request["checksum"]:
                raise ValueError("input checksum mismatch")
            started = time.perf_counter()
            nlm_search_window = int(request["nlm_search_window"])
            nlm_h = float(request["nlm_h"])
            nlm_h_color = float(request["nlm_h_color"])
            future = pool.submit(
                _process, payload, tuple(request["shape"]),
                nlm_search_window, nlm_h, nlm_h_color,
            )
            samples: list[tuple[float, float]] = []
            proc = psutil.Process()
            proc.cpu_percent(None)
            while not future.done():
                time.sleep(0.25)
                children = proc.children(recursive=True)
                cpu = proc.cpu_percent(None) + sum(child.cpu_percent(None) for child in children if child.is_running())
                samples.append((cpu, psutil.virtual_memory().percent))
                if time.perf_counter() - started > SOCKET_TIMEOUT:
                    future.cancel()
                    raise TimeoutError("worker computation timed out")
            output, shape, digest, compute_seconds = future.result()
            avg_cpu = sum(x[0] for x in samples) / len(samples) if samples else 0.0
            avg_mem = sum(x[1] for x in samples) / len(samples) if samples else psutil.virtual_memory().percent
            send_json(conn, {"ok": True, "task_id": request["task_id"], "host": socket.gethostname(), "shape": list(shape), "dtype": "uint8", "checksum": digest, "compute_seconds": compute_seconds, "input_receive_seconds": input_receive_seconds, "cpu_percent": avg_cpu, "memory_percent": avg_mem})
            send_bytes(conn, output)
            send_json(conn, {"complete": True})
            print(f"[TASK] {task_label} completed from {master_label}", flush=True)
        except Exception as exc:
            print(f"[ERROR] task={task_label} master={master_label} {type(exc).__name__}: {exc}", flush=True)
            try:
                send_json(conn, {"ok": False, "error": str(exc)})
            except OSError:
                pass
        finally:
            if reserved:
                capacity.release()


def run(bind: str, port: int, processes: int) -> None:
    capacity = threading.BoundedSemaphore(processes)

    def discovery() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp.bind(("", DISCOVERY_PORT))
            while True:
                try:
                    data, addr = udp.recvfrom(1024)
                    if data == b"DISCOVER_UPSCAYL_WORKERS_V1":
                        info = {"host": socket.gethostname(), "port": port, "cores": processes, "protocol": PROTOCOL_VERSION, "operations": ["nlm"], "nlm_task_schema": 4}
                        udp.sendto(__import__("json").dumps(info).encode(), addr)
                except OSError:
                    break

    threading.Thread(target=discovery, daemon=True).start()
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as pool:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((bind, port)); server.listen()
            print(f"[WORKER] {socket.gethostname()} listening {bind}:{port}, processes={processes}", flush=True)
            while True:
                conn, addr = server.accept()
                threading.Thread(target=handle, args=(conn, addr, pool, capacity), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent distributed image worker")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    run(args.bind, args.port, max(1, args.processes))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
