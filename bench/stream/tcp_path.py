"""Measure the laptop -> Pi path without iperf3: TCP bulk throughput (laptop sends
to a sink on the Pi) and ping RTT with 1400-byte packets.

  python3 tcp_path.py sink PORT                       # on the Pi
  python3 tcp_path.py send HOST PORT MB [BIND_ADDR]   # on the laptop
"""
import json
import socket
import sys
import time


def sink(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    c, _ = s.accept()
    n, t0 = 0, None
    buf = bytearray(1 << 20)
    while True:
        k = c.recv_into(buf)
        if not k:
            break
        if t0 is None:
            t0 = time.monotonic()
        n += k
    dt = time.monotonic() - t0
    c.sendall(json.dumps({"bytes": n, "seconds": dt, "mbit_s": 8 * n / dt / 1e6}).encode())
    c.close()


def send(host, port, mb, bind=None):
    s = socket.socket()
    if bind:
        s.bind((bind, 0))
    s.connect((host, port))
    local = s.getsockname()
    chunk = b"\0" * (1 << 20)
    t0 = time.monotonic()
    for _ in range(mb):
        s.sendall(chunk)
    s.shutdown(socket.SHUT_WR)
    r = json.loads(s.recv(4096).decode())
    r["sender_seconds"] = time.monotonic() - t0
    r["local"] = f"{local[0]}:{local[1]}"
    print(json.dumps(r))


if __name__ == "__main__":
    if sys.argv[1] == "sink":
        sink(int(sys.argv[2]))
    else:
        send(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)
