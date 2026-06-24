# Compression Evaluation

This directory contains preliminary edge compression comparison code and sample results.

## Contents

- `code/compression-comparison-rf.py`: comparison script for LZ4, Gzip, and Gorilla-style compression.
- `sample-results/compression-rf-comparison.csv`: sample result table.
- `sample-results/best-effective-throughput.svg`: sample effective throughput chart.
- `compression-report.md`: current-stage compression report.

The current work evaluates compression independently. Integrated protocol selection plus compression plus caching ablation is still future work.
