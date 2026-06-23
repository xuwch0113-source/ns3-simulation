# NS-3 Simulation Source and Run Guide

This directory contains the NS-3 TCP simulation source used in the current stage.

## Contents

- `source/tcp-exp.cc`: point-to-point TCP simulation for TCP CUBIC and TCP BBR.
- `run-guide/ns3-run-guide.md`: topology, parameters, commands, and QUIC dependency notes.

## Note

QUIC BBR is evaluated through an external ns-3.32 QUIC module. The external module is not bundled in this submission package.
