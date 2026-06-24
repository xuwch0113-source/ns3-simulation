#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Unified runner for reliable transport experiments in this ns-3-dev tree and
# the external ns-3.32 QUIC experiment.

import argparse
import csv
import os
import random
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ns3")


def find_ns3_root():
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ns3").exists() and (candidate / "scratch").exists():
            return candidate
    return Path.cwd().resolve()


ROOT = find_ns3_root()
QUIC_ROOT = Path("/Users/xuweicheng/ns-allinone-3.32/ns-3.32")
TCP_EXP = ROOT / "build" / "scratch" / "ns3-dev-tcp-exp-default"
TCP_SOURCE = ROOT / "scratch" / "tcp-exp.cc"
QUIC_EXP = QUIC_ROOT / "build" / "contrib" / "quic" / "examples" / "ns3.32-quic-variants-comparison-bulksend-debug"

THROUGHPUT_RE = re.compile(r"averageThroughput=([0-9.]+) Mbit/s")

SCENARIOS = {
    "china-us": 237.886,
    "hangzhou-lanzhou": 5.853,
}

TCP_TYPES = {
    "TCP_CUBIC": "ns3::TcpCubic",
    "TCP_BBR": "ns3::TcpBbr",
}


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_loss_range(value):
    start, stop, step = [float(item.strip()) for item in value.split(":", 2)]
    if step <= 0 or stop < start:
        raise ValueError("--loss-range must use start:stop:positive_step")
    values = []
    current = start
    epsilon = step / 1000.0
    while current <= stop + epsilon:
        values.append(round(current, 6))
        current += step
    return values


def parse_interval_ranges(value):
    ranges = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Send interval ranges must use min_ms:max_ms format, "
                f"for example 1:10. Got: {item}"
            )
        low, high = [float(part.strip()) for part in item.split(":", 1)]
        if low <= 0 or high <= 0 or high < low:
            raise ValueError(f"Invalid send interval range: {item}")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("At least one send interval range is required.")
    return ranges


def parse_random_loss_spec(value, seed):
    count_text, low_text, high_text = value.split(":", 2)
    count = int(count_text)
    low = float(low_text)
    high = float(high_text)
    if count <= 0 or low < 0 or high < low:
        raise ValueError("--random-loss-count must use count:min:max")
    rng = random.Random(seed)
    return [round(rng.uniform(low, high), 6) for _ in range(count)]


def parse_scenarios(value):
    scenarios = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Scenario entries must use name:rtt_ms format, "
                f"for example rtt-100:100. Got: {item}"
            )
        name, rtt = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Scenario name cannot be empty in entry: {item}")
        scenarios[name] = float(rtt.strip())
    if not scenarios:
        raise ValueError("At least one scenario must be provided.")
    return scenarios


def run_command(cmd, cwd, timeout):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.stdout


def parse_throughput(output):
    match = THROUGHPUT_RE.search(output)
    if not match:
        raise RuntimeError(f"Cannot parse throughput from output:\n{output}")
    return float(match.group(1))


def build_ns3_dev():
    if not TCP_EXP.exists() or TCP_SOURCE.stat().st_mtime > TCP_EXP.stat().st_mtime:
        run_command(["./ns3", "build", "tcp-exp"], ROOT, timeout=300)


def build_quic():
    if not QUIC_ROOT.exists():
        raise FileNotFoundError(f"QUIC ns-3.32 directory not found: {QUIC_ROOT}")
    if not QUIC_EXP.exists():
        run_command(
            ["/usr/bin/python3", "./waf", "configure", "--enable-examples", "--disable-python", "--disable-werror"],
            QUIC_ROOT,
            timeout=120,
        )
        run_command(["/usr/bin/python3", "./waf", "build"], QUIC_ROOT, timeout=600)


def run_tcp(protocol,
            rtt_ms,
            capacity_mbps,
            loss_rate,
            duration,
            traffic_mode,
            min_interval_ms,
            max_interval_ms,
            packet_size,
            file_size_mb):
    bottleneck_delay_ms = rtt_ms / 2.0
    stop_time = 1.0 + duration
    max_application_bytes = int(file_size_mb * 1_000_000) if file_size_mb > 0 else 0
    args = [
        f"--tcpType={TCP_TYPES[protocol]}",
        "--accessDelay=0ms",
        "--egressDelay=0ms",
        f"--bottleneckRate={capacity_mbps:g}Mbps",
        f"--bottleneckDelay={bottleneck_delay_ms:.6f}ms",
        f"--bottleneckLoss={loss_rate:g}",
        f"--trafficMode={traffic_mode}",
        f"--segmentSize={packet_size}",
        f"--minSendIntervalMs={min_interval_ms:g}",
        f"--maxSendIntervalMs={max_interval_ms:g}",
        f"--maxApplicationBytes={max_application_bytes}",
        f"--stopTime={stop_time:g}",
    ]
    output = run_command([str(TCP_EXP)] + args, ROOT, timeout=max(90, int(duration * 12)))
    command = f"cd {ROOT} && {' '.join([str(TCP_EXP)] + args)}"
    return parse_throughput(output), command


def run_quic(rtt_ms, capacity_mbps, loss_rate, duration, file_size_mb):
    one_way_delay_ms = rtt_ms / 2.0
    args = [
        "--transport_prot=QuicBbr",
        f"--bandwidth={capacity_mbps:g}Mbps",
        f"--delay={one_way_delay_ms:.6f}ms",
        "--access_delay=0ms",
        f"--error_p={loss_rate:g}",
        f"--data={file_size_mb:g}",
        f"--duration={duration:g}",
    ]
    output = run_command(
        [str(QUIC_EXP)] + args,
        QUIC_ROOT,
        timeout=max(120, int(duration * 30)),
    )
    command = f"cd {QUIC_ROOT} && {' '.join([str(QUIC_EXP)] + args)}"
    return parse_throughput(output), command


def write_csv(path, rows):
    fieldnames = [
        "scenario",
        "protocol",
        "rtt_ms",
        "capacity_mbps",
        "packet_loss_rate",
        "traffic_mode",
        "packet_size_bytes",
        "min_send_interval_ms",
        "max_send_interval_ms",
        "mean_send_interval_ms",
        "offered_load_mbps",
        "file_size_mb",
        "estimated_file_transfer_time_s",
        "duration_s",
        "throughput_mbps",
        "status",
        "command",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    with path.open("w") as stream:
        stream.write("# Combined Experiment Summary\n\n")
        stream.write("All commands were launched from the unified runner:\n\n")
        stream.write("```bash\n")
        stream.write("cd /Users/xuweicheng/ns-3-dev\n")
        stream.write("python3 scratch/run-all-protocol-evaluations.py\n")
        stream.write("```\n\n")
        for row in rows:
            throughput = row["throughput_mbps"]
            if throughput == "":
                throughput = "failed"
            else:
                throughput = f"{throughput:.6f} Mbit/s"
            stream.write(
                f"- {row['scenario']} {row['protocol']}: "
                f"RTT={row['rtt_ms']:.3f} ms, "
                f"capacity={row['capacity_mbps']:g} Mbps, "
                f"loss={row['packet_loss_rate']:g}, "
                f"traffic={row['traffic_mode']}, "
                f"sendInterval={row['min_send_interval_ms']:g}-{row['max_send_interval_ms']:g} ms, "
                f"fileSize={row['file_size_mb']:g} MB, "
                f"estimatedFileTime={row['estimated_file_transfer_time_s'] if row['estimated_file_transfer_time_s'] != '' else 'n/a'} s, "
                f"throughput={throughput}\n"
            )


def plot_results(path, rows):
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row["throughput_mbps"] != ""]
    if not ok_rows:
        return

    labels = [
        f"{row['scenario']}\n{row['protocol']}\n{row['capacity_mbps']:g}M loss={row['packet_loss_rate']:g}"
        for row in ok_rows
    ]
    values = [row["throughput_mbps"] for row in ok_rows]

    plt.figure(figsize=(max(10, len(labels) * 0.7), 5.5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Throughput (Mbit/s)")
    plt.title("Reliable transport throughput under poor network conditions")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", default="0.5,1,2")
    parser.add_argument("--losses", default="0.01,0.03")
    parser.add_argument(
        "--loss-range",
        default="",
        help="Continuous loss range in start:stop:step format, for example 0.001:0.08:0.002",
    )
    parser.add_argument(
        "--random-loss-count",
        default="",
        help="Additional random loss samples in count:min:max format, for example 20:0.001:0.12",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--traffic-mode",
        choices=["bulk", "random-interval"],
        default="bulk",
    )
    parser.add_argument(
        "--send-interval-ranges",
        default="1:10",
        help="Comma-separated min_ms:max_ms ranges, for example 1:5,5:20,20:80",
    )
    parser.add_argument("--packet-size", type=int, default=1448)
    parser.add_argument(
        "--file-size-mb",
        type=float,
        default=0.0,
        help=(
            "Fixed file size used to estimate file transfer time. "
            "Also passed to TCP maxApplicationBytes and QUIC --data when > 0."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Comma-separated scenario list in name:rtt_ms format. "
            "Default uses built-in china-us and hangzhou-lanzhou."
        ),
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output-dir", default="scratch/protocol-evaluation-results")
    parser.add_argument("--skip-quic", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    capacities = parse_list(args.capacities, float)
    losses = parse_loss_range(args.loss_range) if args.loss_range else parse_list(args.losses, float)
    if args.random_loss_count:
        losses.extend(parse_random_loss_spec(args.random_loss_count, args.seed))
    losses = sorted(set(round(loss, 6) for loss in losses))
    interval_ranges = parse_interval_ranges(args.send_interval_ranges)
    scenarios = parse_scenarios(args.scenarios) if args.scenarios else SCENARIOS
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building ns-3-dev TCP programs...")
    build_ns3_dev()

    if not args.skip_quic:
        print("Building external ns-3.32 QUIC program...")
        build_quic()

    rows = []
    for scenario, rtt_ms in scenarios.items():
        for capacity in capacities:
            for loss in losses:
                for min_interval_ms, max_interval_ms in interval_ranges:
                    mean_interval_ms = (min_interval_ms + max_interval_ms) / 2.0
                    offered_load_mbps = args.packet_size * 8.0 / (mean_interval_ms / 1000.0) / 1_000_000.0
                    jobs = [
                        (
                            "TCP_CUBIC",
                            lambda rtt=rtt_ms, cap=capacity, ls=loss, low=min_interval_ms, high=max_interval_ms: run_tcp(
                                "TCP_CUBIC",
                                rtt,
                                cap,
                                ls,
                                args.duration,
                                args.traffic_mode,
                                low,
                                high,
                                args.packet_size,
                                args.file_size_mb,
                            ),
                        ),
                        (
                            "TCP_BBR",
                            lambda rtt=rtt_ms, cap=capacity, ls=loss, low=min_interval_ms, high=max_interval_ms: run_tcp(
                                "TCP_BBR",
                                rtt,
                                cap,
                                ls,
                                args.duration,
                                args.traffic_mode,
                                low,
                                high,
                                args.packet_size,
                                args.file_size_mb,
                            ),
                        )
                    ]
                    if not args.skip_quic:
                        jobs.append(
                            (
                                "QUIC_BBR",
                                lambda rtt=rtt_ms, cap=capacity, ls=loss: run_quic(
                                    rtt,
                                    cap,
                                    ls,
                                    args.duration,
                                    args.file_size_mb,
                                ),
                            )
                        )

                    for protocol, job in jobs:
                        row = {
                            "scenario": scenario,
                            "protocol": protocol,
                            "rtt_ms": rtt_ms,
                            "capacity_mbps": capacity,
                            "packet_loss_rate": loss,
                            "traffic_mode": args.traffic_mode if protocol != "QUIC_BBR" else "quic-bulksend",
                            "packet_size_bytes": args.packet_size,
                            "min_send_interval_ms": min_interval_ms,
                            "max_send_interval_ms": max_interval_ms,
                            "mean_send_interval_ms": mean_interval_ms,
                            "offered_load_mbps": offered_load_mbps,
                            "file_size_mb": args.file_size_mb,
                            "estimated_file_transfer_time_s": "",
                            "duration_s": args.duration,
                            "throughput_mbps": "",
                            "status": "ok",
                            "command": "",
                        }
                        try:
                            throughput, command = job()
                            row["throughput_mbps"] = throughput
                            if args.file_size_mb > 0 and throughput > 0:
                                row["estimated_file_transfer_time_s"] = (
                                    args.file_size_mb * 8.0 / throughput
                                )
                            row["command"] = command
                            print(
                                f"{scenario} {protocol}: capacity={capacity:g} Mbps, "
                                f"loss={loss:g}, interval={min_interval_ms:g}-{max_interval_ms:g} ms, "
                                f"throughput={throughput:.6f} Mbit/s",
                                flush=True,
                            )
                        except Exception as exc:
                            row["status"] = f"failed: {exc}"
                            print(
                                f"{scenario} {protocol}: capacity={capacity:g} Mbps, "
                                f"loss={loss:g}, interval={min_interval_ms:g}-{max_interval_ms:g} ms, "
                                f"failed: {exc}",
                                flush=True,
                            )
                        rows.append(row)

    write_csv(output_dir / "protocol-evaluation-results.csv", rows)
    write_summary(output_dir / "result-summary.md", rows)
    if not args.no_plot:
        plot_results(output_dir / "combined-throughput-plot.png", rows)

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
