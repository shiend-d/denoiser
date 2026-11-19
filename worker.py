from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import socket
import time

import numpy as np

from config import DEFAULT_PORT, DISCOVERY_PORT, PROTOCOL_VERSION, SOCKET_TIMEOUT
from image_pipeline import checksum, process_slice
from protocol import recv_bytes, recv_json, send_bytes, send_json


def _process(payload: bytes, shape: tuple[int, ...], scale: float) -> tuple[bytes, tuple[int, ...], str]:
    image = np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
    result = process_slice(image, scale)
    result_bytes = np.ascontiguousarray(result).tobytes()
    return result_bytes, result.shape, checksum(result)


def handle(conn: socket.socket, address: tuple[str, int], pool: concurrent.futures.ProcessPoolExecutor) -> None:
    with conn:
        conn.settimeout(SOCKET_TIMEOUT)
        try:
            request = recv_json(conn)
            if request.get("action") == "info":
                send_json(conn, {"ok": True, "host": socket.gethostname(), "cores": multiprocessing.cpu_count(), "slots": multiprocessing.cpu_count()})
                return
            if request.get("version") != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol version")
            payload = recv_bytes(conn)
            if checksum(np.frombuffer(payload, dtype=np.uint8)) != request["checksum"]:
                raise ValueError("input checksum mismatch")
            started = time.perf_counter()
            future = pool.submit(_process, payload, tuple(request["shape"]), float(request["scale"]))
            output, shape, digest = future.result(timeout=SOCKET_TIMEOUT)
            send_json(conn, {"ok": True, "task_id": request["task_id"], "shape": list(shape), "dtype": "uint8", "checksum": digest, "compute_seconds": time.perf_counter() - started})
            send_bytes(conn, output)
            print(f"[WORKER] task={request['task_id']} master={address[0]} input={len(payload)}B output={len(output)}B checksum={digest}", flush=True)
        except Exception as exc:
            print(f"[WORKER][ERROR] master={address[0]} error={type(exc).__name__}: {exc}", flush=True)
            try:
                send_json(conn, {"ok": False, "error": str(exc)})
            except OSError:
                pass


def run(bind: str, port: int, processes: int) -> None:
    import threading

    def discovery() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp.bind(("", DISCOVERY_PORT))
            while True:
                try:
                    data, addr = udp.recvfrom(1024)
                    if data == b"DISCOVER_UPSCAYL_WORKERS_V1":
                        info = {"host": socket.gethostname(), "port": port, "cores": processes, "protocol": PROTOCOL_VERSION}
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
                threading.Thread(target=handle, args=(conn, addr, pool), daemon=True).start()


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
