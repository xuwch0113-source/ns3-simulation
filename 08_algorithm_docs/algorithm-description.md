# Algorithm Description

## Adaptive Protocol Selection Task

Input features:

- RTT
- Capacity
- Packet loss rate
- Dynamic link state
- Packet size statistics
- Send interval statistics
- Offered load
- Ocean data type
- Previous-window protocol state

Output action:

- TCP CUBIC
- TCP BBR
- QUIC BBR

## Label Definition

For each training sample, the label `best_protocol` is the protocol with the shortest estimated file transfer time among TCP CUBIC, TCP BBR, and QUIC BBR.

## Models

Evaluated models:

- Random Forest
- LightGBM
- XGBoost
- TinyMLP
- CatBoost
- Linear SVM
- Contextual Bandit Ridge

Tree-based models include Random Forest, LightGBM, XGBoost, and CatBoost. TinyMLP is a lightweight neural network. Contextual Bandit estimates the reward of each protocol action and selects the highest-reward action.

## Current Interpretation

- Random Forest has the best Macro F1 and is more balanced across protocol classes.
- LightGBM has the best Macro AUC and ranks protocol quality well.
- CatBoost has the smallest Oracle gap and is closest to the throughput-oriented objective.

## Packet Size and Send Interval

Packet size affects protocol selection mainly through offered load:

```text
offered_load_mbps = packet_size_mean * 8 / interval_mean
```

Larger packets and shorter intervals increase traffic pressure, which can amplify protocol differences under poor link conditions.
