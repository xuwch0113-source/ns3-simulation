# System Design

## Scenario

The system models ocean observation data transmission from an underwater edge node to an onshore server:

```text
Underwater sensor / edge node -> submarine cable link -> onshore server
```

At the current stage, the sensor and the edge node are treated as colocated. The main focus is the submarine cable link between the edge node and the onshore server.

## Optimization Directions

1. Adaptive protocol selection: dynamically select TCP CUBIC, TCP BBR, or QUIC BBR according to link and traffic state.
2. Edge real-time compression: compress sensor data before transmission and decompress it onshore.
3. Edge caching: buffer data at the edge node when the cable link degrades.

## Current Focus

The current deliverable mainly covers the first direction: adaptive transport protocol selection. Compression comparison is preliminary. Cache mechanism implementation remains future work.

## Metrics

- Throughput
- File transfer time
- Accuracy
- Macro F1
- Macro AUC
- Oracle gap
- Compression ratio
- Effective throughput
- Future cache data loss rate
