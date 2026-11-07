# Low-Power Ultrasonic Gas Leak Detection with SNNs — Step 1 (data, split, first training)

## Setup
```bash
pip install torch snntorch numpy scipy pandas scikit-learn matplotlib
```

## Run (CPU is fine, ~30 min total)
```bash
python build_dataset.py              # 3000 synthetic clips + 70/15/15 split  (~1 min)
python train.py --model cnn --epochs 40   # baseline                         (~3 min)
python train.py --model snn --epochs 25   # spiking network                   (~25 min)
python evaluate.py                   # metrics + figures in runs/
# real-data check (download GPLA-12 data_v1/data.csv + label.csv first, set DATA path):
python gpla12_experiment.py
```
GPLA-12: https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12 (folder data/data_v1)

## Files
| File | What it does |
|---|---|
| `gasleak/synth.py` | Simulates 1 s, 96 kHz clips: leak hiss (24–40 kHz, pressure dependent, sudden onset), factory noise (machines, gears, impacts), ultrasonic distractors (air bursts, tones). SNR set in 18–46 kHz band. |
| `gasleak/features.py` | 16 ultrasonic bands (1 ms frames) → event encoder: noise-floor tracker + sustained (integrate-and-fire) and onset (delta) spikes = 32 input channels. |
| `gasleak/models.py` | `LeakSNN` (32→LIF64→LIF64→leaky readout) and `CausalCNN` baseline. |
| `build_dataset.py` | Generates data, stratified 70/15/15 split → `data/synth/meta.csv` (split column). |
| `train.py` | Per-time-step leak/no-leak training (BCE, 30 ms grace after onset). |
| `evaluate.py` | Detection rate, false alarms, detection delay, SNR curves, compute energy. |
| `gpla12_experiment.py` | Same SNN recipe on real GPLA-12 recordings (12 classes). |

## Energy model (estimate, not measurement)
SNN: synaptic ops = Σ spikes × fan-out, 0.9 pJ each (32-bit add, 45 nm, Horowitz 2014).
CNN: MACs per 1 ms frame × 1000, 4.6 pJ each. Front-end filterbank is common to both and **not** included yet (Week 4).
