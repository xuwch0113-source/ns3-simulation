# Compression Evaluation Report

## Goal

The edge node compresses sensor data before transmission, and the onshore side decompresses it after receiving the data. Throughput is calculated using the original uncompressed data size, so compression can improve effective transmission efficiency.

## Compared Methods

- LZ4: fast compression, lower compression ratio.
- Gzip: stronger compression, higher CPU cost.
- Gorilla-style encoding: suitable for time-series sensor data.

## Current Status

The current stage provides a preliminary comparison based on the Random Forest protocol selector. Full integration with adaptive protocol selection and cache control remains future work.
