# Datasets and Generators

This directory contains raw ocean observation samples, dataset generation scripts, and the current large training dataset.

## Contents

- `raw-samples/`: ocean observation sample files with English filenames. One large weather source is also archived as `.txt.gz`.
- `generators/build-expanded-learning-data.py`: early expanded learning dataset script.
- `generators/prepare-large-training-dataset.py`: current large dataset generation script.
- `generators/plot-training-data-distribution.py`: data distribution plotting script.
- `training-data/adaptive-protocol-large-training-dataset.csv.gz`: compressed 120000-sample training dataset.
- `training-data/adaptive-protocol-training-sample.csv`: small plain CSV subset for quick inspection.
- `training-data/ocean-sample-profiles.csv`: extracted ocean sample profiles.

## Dataset Note

The 120000 training rows are augmented samples generated from measured protocol-evaluation anchors and ocean observation sample profiles. They are not 120000 fresh NS-3 runs. Final conclusions should still be validated with representative NS-3 simulations.
