from __future__ import annotations

import json
import socket
import struct
from typing import Any

HEADER = struct.Struct("!Q")

def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        data.extend(chunk)
    return bytes(data)

def send_bytes(sock: socket.socket, data: bytes) -> None:
    sock.sendall(HEADER.pack(len(data)))
    sock.sendall(data)

def recv_bytes(sock: socket.socket) -> bytes:
    size = HEADER.unpack(recv_exact(sock, HEADER.size))[0]
    if size > 2**32:
        raise ValueError(f"payload too large: {size} bytes")
    return recv_exact(sock, size)

def send_json(sock: socket.socket, value: dict[str, Any]) -> None:
    send_bytes(sock, json.dumps(value, separators=(",", ":")).encode())

def recv_json(sock: socket.socket) -> dict[str, Any]:
    value = json.loads(recv_bytes(sock).decode())
    if not isinstance(value, dict):
        raise ValueError("JSON message must be an object")
    return value
