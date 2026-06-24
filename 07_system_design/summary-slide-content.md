# Summary Slide Content

## Title

Stage Summary: Adaptive Data Transport Protocol Selection Under Dynamic Links

## Slide Text

This stage focuses on the submarine cable link between an underwater edge node and an onshore server. We built dynamic link scenarios, evaluated representative reliable transport protocols, generated a large training dataset, and compared lightweight machine learning models for adaptive protocol selection.

## Key Results

- Large training dataset: 120000 augmented samples.
- Best Macro F1: Random Forest, 0.700.
- Best Macro AUC: LightGBM, 0.950.
- Smallest Oracle gap: CatBoost, about 0.021 Mbit/s.

## Speaker Notes

This page summarizes the current first-stage work. We first built a dynamic submarine cable link scenario and then evaluated TCP CUBIC, TCP BBR, and QUIC BBR under different link states. Based on these evaluation results, we generated 120000 training samples that include RTT, capacity, packet loss rate, packet size, send interval, and ocean observation data type.

For model selection, we compared several lightweight models rather than relying only on a hand-written rule tree. The current results show that adaptive learned protocol selection is better than fixed static protocol policies. Random Forest is more balanced in classification, LightGBM has the best ranking ability, and CatBoost is closest to the oracle throughput objective.
