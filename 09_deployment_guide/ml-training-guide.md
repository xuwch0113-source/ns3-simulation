# Lightweight Model Training Guide

## Models

The training script evaluates:

- LightGBM
- XGBoost
- Random Forest
- TinyMLP
- CatBoost
- Linear SVM
- Contextual Bandit Ridge

## Recommended Command

```bash
python3 08_algorithm_docs/train-lightweight-protocol-selector.py \
  --max-rows=30000 \
  --fast \
  --output-dir=08_algorithm_docs/model-results
```

## Current Result Summary

Training samples: 22500  
Test samples: 7500

| Model | Accuracy | Macro F1 | Macro AUC | Avg selected throughput | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.801 | 0.700 | 0.930 | 0.363 | 0.035 |
| LightGBM | 0.841 | 0.695 | 0.950 | 0.375 | 0.022 |
| TinyMLP | 0.830 | 0.687 | 0.941 | 0.375 | 0.022 |
| XGBoost | 0.833 | 0.638 | 0.945 | 0.376 | 0.021 |
| CatBoost | 0.826 | 0.570 | 0.938 | 0.376 | 0.021 |

Interpretation:

- Random Forest is best by Macro F1.
- LightGBM is best by Macro AUC.
- CatBoost is closest to the oracle throughput objective.
