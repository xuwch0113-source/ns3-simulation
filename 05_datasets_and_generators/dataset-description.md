# Dataset Description

## Data Sources

- Protocol evaluation anchors from NS-3 and QUIC experiments.
- Ocean observation raw sample profiles.
- Dynamic link state augmentation.

## Main Features

- RTT
- Capacity
- Packet loss rate
- Dynamic link state
- Packet size statistics
- Send interval statistics
- Offered load
- Ocean data type
- Previous-window protocol state

## Label

The supervised label is `best_protocol`, selected by the shortest estimated file transfer time among TCP CUBIC, TCP BBR, and QUIC BBR.

## Packet Size and Offered Load

Random packet size affects the model primarily through offered load:

```text
offered_load_mbps = packet_size_mean * 8 / interval_mean
```

Larger packets and shorter send intervals increase traffic pressure, which can change the best protocol under poor link conditions.
