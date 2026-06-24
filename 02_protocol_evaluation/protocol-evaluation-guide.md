# Protocol Evaluation Guide

## Full Protocol Evaluation

Copy these scripts into the NS-3 `scratch/` directory first, then run the commands from the NS-3 project root.

```bash
python3 scratch/run-all-protocol-evaluations.py \
  --scenarios=rtt-80:80,rtt-150:150,rtt-237:237.886 \
  --capacities=0.2,0.3,0.5,1,2,3 \
  --losses=0.001,0.02,0.05,0.1,0.15 \
  --traffic-mode=bulk \
  --send-interval-ranges=1:10 \
  --file-size-mb=10 \
  --duration=3 \
  --output-dir=scratch/protocol-evaluation-file-time \
  --no-plot
```

Plotting:

```bash
python3 scratch/plot-protocol-evaluation.py \
  --input=scratch/protocol-evaluation-file-time/protocol-evaluation-results.csv
```

## Dynamic Link Evaluation

```bash
python3 scratch/dynamic-link-adaptive-protocol.py \
  --scenario-mode=progressive-bad-recover \
  --phase-count=18 \
  --phase-duration=3 \
  --normal-ratio=0.1 \
  --severe-ratio=0.55 \
  --traffic-mode=random-interval \
  --send-interval-ranges=1:8,8:30,30:120 \
  --output-dir=scratch/dynamic-link-results
```

## Sample Results

The `sample-results/` directory contains representative CSV files and figures from the current stage.
