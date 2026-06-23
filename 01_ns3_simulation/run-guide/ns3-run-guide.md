# NS-3 Run Guide

## Topology

The simulation abstracts the submarine cable link as a point-to-point connection:

```text
Underwater edge node -> submarine cable link -> onshore server
```

Configurable link parameters:

- RTT: `--rttMs`
- Capacity: `--capacityMbps`
- Packet loss rate: `--lossRate`
- Traffic mode: `bulk` or `random-interval`

## Build and Run

Place `source/tcp-exp.cc` in the NS-3 `scratch/` directory, then run:

```bash
./ns3 build scratch/tcp-exp
./ns3 run "scratch/tcp-exp --tcpType=TcpCubic --rttMs=150 --capacityMbps=2 --lossRate=0.03 --duration=5"
./ns3 run "scratch/tcp-exp --tcpType=TcpBbr --rttMs=150 --capacityMbps=2 --lossRate=0.03 --duration=5"
```

Random interval traffic example:

```bash
./ns3 run "scratch/tcp-exp --tcpType=TcpBbr --rttMs=150 --capacityMbps=2 --lossRate=0.03 --trafficMode=random-interval --minSendIntervalMs=1 --maxSendIntervalMs=30 --duration=5"
```

## QUIC

QUIC BBR is launched by `02_protocol_evaluation/scripts/run-all-protocol-evaluations.py` through an external ns-3.32 QUIC environment.
