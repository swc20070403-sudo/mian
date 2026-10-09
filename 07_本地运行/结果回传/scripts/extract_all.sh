#!/usr/bin/env bash
# Frozen extractor on each modified pickle (system Python 3.10 with pandas + PyWavelets).
# The extractor's final LFP-count assertions pass here (same rows as the original LFP file).
cd "$(dirname "$0")/.."
P=../../scripts/extract_lfp_features.py
for v in noise_1p0mV noise_2p0mV noise_0p5mV offset_plus100mV; do
  python "$P" --pulse-data "data/LFP_slowpulse_$v.pkl" --output-dir "features_$v" > "logs/extract_$v.txt" 2>&1 &
done
wait
echo all-done
