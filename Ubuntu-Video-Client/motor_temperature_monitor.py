#!/usr/bin/env python3
"""Standalone G1 motor temperature warning window.

Subscribes to Sonic deploy's g1_debug ZMQ topic and displays only the maximum
motor temperature and motors above the warning threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import queue
import sys
import threading
import time


MOTOR_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


@dataclass(frozen=True)
class MotorTemp:
    index: int
    name: str
    winding: int
    driver: int

    @property
    def max_temp(self) -> int:
        return max(self.winding, self.driver)


@dataclass(frozen=True)
class TemperatureSnapshot:
    timestamp: float
    motors: tuple[MotorTemp, ...] = ()
    error: str = ""

    @property
    def max_motor(self) -> MotorTemp | None:
        if not self.motors:
            return None
        return max(self.motors, key=lambda motor: motor.max_temp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a standalone G1 motor temperature warning window."
    )
    parser.add_argument("--host", required=True, help="Orin/Sonic deploy IP address")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--topic", default="g1_debug")
    parser.add_argument("--warning", type=int, default=90)
    parser.add_argument("--critical", type=int, default=100)
    parser.add_argument("--stale-sec", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--topmost", action="store_true")
    args = parser.parse_args()

    if args.critical < args.warning:
        parser.error("--critical must be >= --warning")
    if args.stale_sec <= 0:
        parser.error("--stale-sec must be > 0")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    return args


def decode_motor_temperatures(values: object) -> tuple[MotorTemp, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("motor_temperature is missing or not an array")
    expected = len(MOTOR_NAMES) * 2
    if len(values) < expected:
        raise ValueError(f"motor_temperature has {len(values)} values; expected {expected}")

    motors = []
    for index, name in enumerate(MOTOR_NAMES):
        winding = int(values[index * 2])
        driver = int(values[index * 2 + 1])
        motors.append(MotorTemp(index=index, name=name, winding=winding, driver=driver))
    return tuple(motors)


def classify_snapshot(
    snapshot: TemperatureSnapshot,
    args: argparse.Namespace,
) -> tuple[str, str, list[MotorTemp]]:
    now = time.monotonic()
    age = now - snapshot.timestamp if snapshot.timestamp else float("inf")

    if snapshot.error:
        return "ERROR", snapshot.error, []
    if not snapshot.timestamp or age > args.stale_sec:
        return "STALE", "Waiting for telemetry...", []

    max_motor = snapshot.max_motor
    if max_motor is None:
        return "STALE", "No motor_temperature data", []

    high_motors = sorted(
        (
            motor
            for motor in snapshot.motors
            if motor.max_temp >= args.warning
        ),
        key=lambda motor: motor.max_temp,
        reverse=True,
    )

    if max_motor.max_temp >= args.critical:
        status = "CRITICAL"
    elif max_motor.max_temp >= args.warning:
        status = "WARNING"
    else:
        status = "OK"

    summary = (
        f"Max: {max_motor.max_temp} C  {max_motor.name}  "
        f"winding={max_motor.winding} driver={max_motor.driver}"
    )
    return status, summary, high_motors


def telemetry_loop(
    args: argparse.Namespace,
    snapshot_queue: queue.Queue[TemperatureSnapshot],
    stop_event: threading.Event,
) -> None:
    try:
        import msgpack  # type: ignore
        import zmq  # type: ignore
    except Exception as exc:
        snapshot_queue.put(
            TemperatureSnapshot(timestamp=time.monotonic(), error=f"dependency error: {exc}")
        )
        return

    context = zmq.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket_sub.setsockopt(zmq.CONFLATE, 1)
    socket_sub.setsockopt(zmq.RCVTIMEO, 200)
    endpoint = f"tcp://{args.host}:{args.port}"

    try:
        socket_sub.connect(endpoint)
        print(f"[temp] subscribing {endpoint} topic={args.topic}", file=sys.stderr)
        topic_len = len(args.topic.encode("utf-8"))

        while not stop_event.is_set():
            try:
                raw = socket_sub.recv()
            except zmq.Again:
                continue
            except Exception as exc:
                snapshot_queue.put(
                    TemperatureSnapshot(timestamp=time.monotonic(), error=str(exc))
                )
                time.sleep(0.2)
                continue

            try:
                payload = raw[topic_len:]
                data = msgpack.unpackb(payload, raw=False)
                motors = decode_motor_temperatures(data.get("motor_temperature"))
                snapshot_queue.put(
                    TemperatureSnapshot(timestamp=time.monotonic(), motors=motors)
                )
            except Exception as exc:
                snapshot_queue.put(
                    TemperatureSnapshot(
                        timestamp=time.monotonic(),
                        error=f"decode error: {exc}",
                    )
                )
    finally:
        socket_sub.close(linger=0)
        context.term()


def drain_latest_snapshot(
    snapshot_queue: queue.Queue[TemperatureSnapshot],
    current: TemperatureSnapshot,
) -> TemperatureSnapshot:
    latest = current
    while True:
        try:
            latest = snapshot_queue.get_nowait()
        except queue.Empty:
            return latest


def run_window(args: argparse.Namespace) -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise RuntimeError(f"tkinter unavailable: {exc}") from exc

    stop_event = threading.Event()
    snapshot_queue: queue.Queue[TemperatureSnapshot] = queue.Queue(maxsize=32)
    current_snapshot = TemperatureSnapshot(timestamp=0.0)

    telemetry_thread = threading.Thread(
        target=telemetry_loop,
        args=(args, snapshot_queue, stop_event),
        daemon=True,
    )
    telemetry_thread.start()

    root = tk.Tk()
    root.title("G1 Motor Temperature")
    root.geometry("520x260")
    root.configure(bg="#111827")
    if args.topmost:
        root.attributes("-topmost", True)

    status_var = tk.StringVar(value="Motor Temp: STALE")
    summary_var = tk.StringVar(value="Waiting for telemetry...")
    high_var = tk.StringVar(value="High motors: none")

    status_label = tk.Label(
        root,
        textvariable=status_var,
        font=("TkDefaultFont", 22, "bold"),
        fg="#f9fafb",
        bg="#4b5563",
        padx=12,
        pady=10,
    )
    status_label.pack(fill="x", padx=12, pady=(12, 8))

    summary_label = tk.Label(
        root,
        textvariable=summary_var,
        font=("TkDefaultFont", 14),
        fg="#e5e7eb",
        bg="#111827",
        anchor="w",
        justify="left",
    )
    summary_label.pack(fill="x", padx=12, pady=(0, 8))

    high_label = tk.Label(
        root,
        textvariable=high_var,
        font=("TkFixedFont", 12),
        fg="#e5e7eb",
        bg="#111827",
        anchor="nw",
        justify="left",
    )
    high_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def close() -> None:
        stop_event.set()
        root.destroy()

    def refresh() -> None:
        nonlocal current_snapshot
        current_snapshot = drain_latest_snapshot(snapshot_queue, current_snapshot)
        status, summary, high_motors = classify_snapshot(current_snapshot, args)

        status_var.set(f"Motor Temp: {status}")
        summary_var.set(summary)

        if status == "OK":
            status_label.configure(bg="#166534", fg="#ecfdf5")
        elif status == "WARNING":
            status_label.configure(bg="#b45309", fg="#fff7ed")
        elif status == "CRITICAL":
            flash_on = int(time.monotonic() * 2) % 2 == 0
            status_label.configure(
                bg="#dc2626" if flash_on else "#7f1d1d",
                fg="#ffffff",
            )
        else:
            status_label.configure(bg="#4b5563", fg="#f9fafb")

        lines = []
        for motor in high_motors[: args.top_k]:
            lines.append(
                f"{motor.name:<22} {motor.max_temp:>3} C  "
                f"winding={motor.winding:>3} driver={motor.driver:>3}"
            )
        if lines:
            lines.insert(0, "High motors:")
        else:
            lines.append("High motors: none")
        high_var.set("\n".join(lines))

        if not stop_event.is_set():
            root.after(250, refresh)

    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    root.mainloop()
    stop_event.set()
    telemetry_thread.join(timeout=1.0)


def main() -> int:
    args = parse_args()
    try:
        run_window(args)
        return 0
    except KeyboardInterrupt:
        print("\n[temp] interrupted", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"[temp] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
