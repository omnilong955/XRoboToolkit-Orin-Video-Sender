#!/usr/bin/env python3
"""Ubuntu client for receiving G1 WBCD H.264 video streams.

The Orin sender transmits each encoded frame as:
  [4-byte big-endian payload length][H.264 payload]

Default mode only receives and displays the stream. Request mode can also send
an XRoboToolkit OPEN_CAMERA command to an OrinVideoSender running with --listen.
"""

from __future__ import annotations

import argparse
from collections import deque
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO


def pack_i32(value: int) -> bytes:
    return struct.pack("<i", value)


def pack_compact_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"XR compact string is too long: {value!r}")
    return bytes([len(raw)]) + raw


def build_camera_request(args: argparse.Namespace) -> bytes:
    payload = bytearray()
    payload += b"\xca\xfe"
    payload += b"\x01"
    payload += pack_i32(args.width)
    payload += pack_i32(args.height)
    payload += pack_i32(args.fps)
    payload += pack_i32(args.bitrate)
    payload += pack_i32(1 if args.hevc else 0)
    payload += pack_i32(args.render_mode)
    payload += pack_i32(args.port)
    payload += pack_compact_string(args.camera)
    payload += pack_compact_string(args.receiver_ip)
    return bytes(payload)


def build_network_command(command: str, data: bytes) -> bytes:
    command_bytes = command.encode("utf-8")
    body = bytearray()
    body += pack_i32(len(command_bytes))
    body += command_bytes
    body += pack_i32(len(data))
    body += data
    return struct.pack(">I", len(body)) + bytes(body)


def send_open_camera(args: argparse.Namespace, stop_event: threading.Event) -> None:
    packet = build_network_command("OPEN_CAMERA", build_camera_request(args))
    endpoint = (args.orin_ip, args.control_port)
    with socket.create_connection(endpoint, timeout=args.connect_timeout) as sock:
        sock.sendall(packet)
        print(
            "[control] sent OPEN_CAMERA to "
            f"{args.orin_ip}:{args.control_port}; "
            f"callback={args.receiver_ip}:{args.port}, "
            f"{args.width}x{args.height}@{args.fps}, bitrate={args.bitrate}",
            file=sys.stderr,
            flush=True,
        )
        while not stop_event.is_set():
            time.sleep(0.2)


def read_exact(conn: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            return None
        chunks += chunk
    return bytes(chunks)


class FfplayDisplay:
    """Feed ffplay from a tiny leaky queue so video display cannot block TCP."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.proc = self._start_ffplay()
        self.queue: deque[bytes] = deque()
        self.cond = threading.Condition()
        self.closed = False
        self.dropped_frames = 0
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def _start_ffplay(self) -> subprocess.Popen[bytes]:
        if shutil.which(self.args.ffplay_bin) is None:
            raise RuntimeError(
                f"{self.args.ffplay_bin!r} not found. Install ffmpeg or use --no-display."
            )

        codec_hint = "hevc" if self.args.hevc else "h264"
        cmd = [
            self.args.ffplay_bin,
            "-hide_banner",
            "-loglevel",
            self.args.ffplay_loglevel,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",
            "-avioflags",
            "direct",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-framerate",
            str(self.args.fps),
            "-sync",
            "ext",
            "-f",
            codec_hint,
            "-i",
            "-",
        ]
        print("[display] " + " ".join(cmd), file=sys.stderr, flush=True)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    def submit(self, payload: bytes) -> None:
        with self.cond:
            if self.closed:
                return

            while len(self.queue) >= self.args.display_queue_size:
                self.queue.popleft()
                self.dropped_frames += 1

            self.queue.append(payload)
            self.cond.notify()

    def _writer_loop(self) -> None:
        while True:
            with self.cond:
                while not self.closed and not self.queue:
                    self.cond.wait()
                if self.closed and not self.queue:
                    break
                payload = self.queue.popleft()

            if self.proc.stdin is None:
                break

            try:
                self.proc.stdin.write(payload)
            except BrokenPipeError:
                print("[display] ffplay closed", file=sys.stderr, flush=True)
                break

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.queue.clear()
            self.cond.notify_all()

        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.terminate()


def start_ffplay(args: argparse.Namespace) -> FfplayDisplay | None:
    if args.no_display:
        return None
    return FfplayDisplay(args)


def open_dump(path: str | None) -> BinaryIO | None:
    if not path:
        return None
    dump_path = Path(path).expanduser()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    return dump_path.open("wb")


def receive_video(args: argparse.Namespace, stop_event: threading.Event) -> int:
    ffplay = start_ffplay(args)
    dump_file = open_dump(args.dump)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.listen_host, args.port))
            server.listen(1)
            print(
                f"[video] listening on {args.listen_host}:{args.port}",
                file=sys.stderr,
                flush=True,
            )
            conn, addr = server.accept()
            print(
                f"[video] connected from {addr[0]}:{addr[1]}",
                file=sys.stderr,
                flush=True,
            )

            with conn:
                total_frames = 0
                total_bytes = 0
                window_frames = 0
                window_bytes = 0
                window_start = time.monotonic()

                while not stop_event.is_set():
                    header = read_exact(conn, 4)
                    if header is None:
                        print("[video] stream closed", file=sys.stderr, flush=True)
                        break

                    payload_size = struct.unpack(">I", header)[0]
                    if payload_size <= 0 or payload_size > args.max_frame_bytes:
                        raise RuntimeError(
                            "invalid payload size "
                            f"{payload_size}; max={args.max_frame_bytes}"
                        )

                    payload = read_exact(conn, payload_size)
                    if payload is None:
                        print("[video] incomplete frame", file=sys.stderr, flush=True)
                        break

                    if dump_file is not None:
                        dump_file.write(payload)

                    if ffplay is not None:
                        ffplay.submit(payload)

                    total_frames += 1
                    total_bytes += payload_size
                    window_frames += 1
                    window_bytes += payload_size

                    now = time.monotonic()
                    if now - window_start >= args.stats_interval:
                        elapsed = now - window_start
                        fps = window_frames / elapsed
                        mbps = window_bytes * 8.0 / elapsed / 1_000_000
                        display_drops = (
                            f", display_drops={ffplay.dropped_frames}"
                            if ffplay is not None
                            else ""
                        )
                        print(
                            f"[stats] fps={fps:.1f}, bitrate={mbps:.2f} Mbps, "
                            f"frames={total_frames}, bytes={total_bytes}"
                            f"{display_drops}",
                            file=sys.stderr,
                            flush=True,
                        )
                        window_frames = 0
                        window_bytes = 0
                        window_start = now

        return 0
    finally:
        stop_event.set()
        if dump_file is not None:
            dump_file.close()
        if ffplay is not None:
            ffplay.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive length-prefixed H.264 video from OrinVideoSender."
    )
    parser.add_argument(
        "--mode",
        choices=("receive", "request"),
        default="receive",
        help="receive: only listen for video; request: also send OPEN_CAMERA",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=8_000_000)
    parser.add_argument("--camera", default="ZED")
    parser.add_argument("--render-mode", type=int, default=2)
    parser.add_argument("--hevc", action="store_true")

    parser.add_argument("--orin-ip", default="")
    parser.add_argument("--control-port", type=int, default=13579)
    parser.add_argument("--receiver-ip", default="")
    parser.add_argument("--connect-timeout", type=float, default=5.0)

    parser.add_argument("--dump", default="", help="Optional raw .h264/.h265 dump path")
    parser.add_argument("--no-display", action="store_true", help="Receive without ffplay")
    parser.add_argument("--ffplay-bin", default="ffplay")
    parser.add_argument("--ffplay-loglevel", default="warning")
    parser.add_argument("--display-queue-size", type=int, default=2)
    parser.add_argument("--stats-interval", type=float, default=5.0)
    parser.add_argument("--max-frame-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()

    if args.mode == "request":
        if not args.orin_ip:
            parser.error("--mode request requires --orin-ip")
        if not args.receiver_ip:
            parser.error("--mode request requires --receiver-ip")
    return args


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    control_thread: threading.Thread | None = None

    if args.mode == "request":
        control_thread = threading.Thread(
            target=send_open_camera,
            args=(args, stop_event),
            daemon=True,
        )
        control_thread.start()

    try:
        return receive_video(args, stop_event)
    except KeyboardInterrupt:
        print("\n[main] interrupted", file=sys.stderr, flush=True)
        stop_event.set()
        return 0
    except Exception as exc:
        stop_event.set()
        print(f"[main] error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if control_thread is not None:
            control_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
