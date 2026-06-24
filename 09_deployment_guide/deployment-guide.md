# Deployment and Run Guide

## 1. NS-3 Setup

From the NS-3 project root:

```bash
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

Copy the NS-3 simulation and protocol-evaluation scripts into the NS-3 `scratch/` directory:

```bash
cp 01_ns3_simulation/source/tcp-exp.cc /path/to/ns-3-dev/scratch/tcp-exp.cc
cp 02_protocol_evaluation/scripts/run-all-protocol-evaluations.py /path/to/ns-3-dev/scratch/run-all-protocol-evaluations.py
cp 02_protocol_evaluation/scripts/plot-protocol-evaluation.py /path/to/ns-3-dev/scratch/plot-protocol-evaluation.py
cp 02_protocol_evaluation/scripts/dynamic-link-adaptive-protocol.py /path/to/ns-3-dev/scratch/dynamic-link-adaptive-protocol.py
```

## 2. Python Dependencies

For the machine-learning and compression scripts in this package, run from the package root and install dependencies locally:

```bash
python3 -m pip install --target python-packages -r 09_deployment_guide/ml-requirements.txt
```

Use the same `python3` executable for dependency installation and for running the scripts. Do not reuse a `python-packages/` directory installed by a different Python version.

## 3. Protocol Evaluation

Run this part from the NS-3 project root after copying the scripts into `scratch/`:

```bash
python3 scratch/run-all-protocol-evaluations.py \
  --scenarios=rtt-80:80,rtt-150:150,rtt-237:237.886 \
  --capacities=0.2,0.3,0.5,1,2,3 \
  --losses=0.001,0.02,0.05,0.1,0.15 \
  --traffic-mode=bulk \
  --file-size-mb=10 \
  --duration=3 \
  --output-dir=scratch/protocol-evaluation-file-time \
  --no-plot
```

## 4. Training Dataset Generation

Run this part from the package root:

```bash
python3 05_datasets_and_generators/generators/prepare-large-training-dataset.py
```

## 5. Lightweight Model Training

```bash
python3 08_algorithm_docs/train-lightweight-protocol-selector.py \
  --max-rows=30000 \
  --fast \
  --output-dir=08_algorithm_docs/model-results
```

## 6. Plot Model Results

```bash
python3 08_algorithm_docs/plot-lightweight-model-results.py
```

## 7. Compression Evaluation

```bash
python3 03_compression_evaluation/code/compression-comparison-rf.py
```

## Notes

- Do not submit local Python dependency folders.
- QUIC BBR requires an external ns-3.32 QUIC environment.
- Some sample data files are raw ocean observation records and may contain original field names.
