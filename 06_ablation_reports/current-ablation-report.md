# Current Ablation Report

## Static Protocol Baselines

The current evaluation includes static TCP CUBIC, static TCP BBR, and static QUIC BBR as baselines.

## Adaptive Protocol Selection

Evaluated lightweight models:

- Random Forest
- LightGBM
- XGBoost
- TinyMLP
- CatBoost
- Linear SVM
- Contextual Bandit Ridge

Current 30000-sample training result:

- Best Macro F1: Random Forest, 0.700.
- Best Macro AUC: LightGBM, 0.950.
- Smallest Oracle gap: CatBoost, about 0.021 Mbit/s.

## Compression

LZ4, Gzip, and Gorilla-style compression have been compared separately.

## Remaining Ablations

- Adaptive protocol selection plus compression.
- Cache mechanism and cache-size analysis.
- Integrated protocol selection plus compression plus caching.
