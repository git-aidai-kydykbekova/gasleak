# Low-Power Ultrasonic Gas Leak Detection with Spiking Neural Networks

**RQ:** Can event-driven spiking neural networks reduce sensor power consumption by 10x during quiescent (no-leak) periods?

Pipeline: ultrasonic audio (96 kHz) → 16-band filterbank → event encoder (spikes) → SNN → alarm.
The baselines are a frame-based CNN that is always on and the same CNN woken up only by encoder events.
Full explanation (in Russian): [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) · Status and to-do list: [PROJECT_STATUS.md](PROJECT_STATUS.md)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python build_dataset.py       # 3000 synthetic 1 s clips + stratified 70/15/15 split          (~1 min)
./run_seeds.sh                # SNN + CNN for seeds 0-4 -> runs/seeds/<seed>/                  (~40 min, CPU)
python evaluate.py            # matched-FA metrics, CIs, DET, energy breakdown -> runs/        (~2 min)
python quiescent.py           # long no-leak streams: false alarms/hour, power                  (~15 min)
python export_firmware.py     # trained SNN + test clips -> C headers for the ESP32-S3
c++ -O2 -std=c++17 firmware/host_test/host_test.cpp -o host_test && ./host_test   # C++ == PyTorch?
python gpla12_experiment.py   # real-data check on GPLA-12 (data/gpla12/data.csv + label.csv)
```
Firmware / online hardware demo (Wokwi ESP32-S3): see [firmware/README.md](firmware/README.md).
GPLA-12: https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12 (folder data/data_v1)

## Files
| File | What it does |
|---|---|
| `gasleak/synth.py` | Simulates 96 kHz audio: leak hiss (24–40 kHz, depends on pressure, sudden onset), factory noise, ultrasonic distractors. 1 s clips, or long no-leak recordings (`n`, `level_db`). SNR is defined in the 18–46 kHz band. |
| `gasleak/features.py` | 16 ultrasonic bands (1 ms frames) → event encoder: noise-floor tracker + sustained (integrate-and-fire) and onset (delta) spikes = 32 input channels. |
| `gasleak/models.py` | `LeakSNN` (32→LIF64→LIF64→leaky readout) and the `CausalCNN` baseline. |
| `gasleak/metrics.py` | Alarm rule (K = 5 ms above threshold), false alarms counted on all leak-free segments (including the part before onset), thresholds for a target FA rate, DET curve. |
| `gasleak/energy.py` | Energy estimate: front-end + encoder + network, for SNN (clock-/event-driven), dense twin, CNN, event-gated CNN. |
| `build_dataset.py` | Generates the data and the stratified split → `data/synth/`. |
| `train.py` | Per-time-step leak/no-leak training (BCE, 30 ms grace after onset); `--seed`, `--out`, `--threads`. |
| `run_seeds.sh` | Trains every model for several seeds. |
| `evaluate.py` | Thresholds at FA ≤ 1 % / 5 % (validation), test metrics mean ± std over seeds, bootstrap 95 % CIs per SNR, DET curves, energy table, RQ ratios. |
| `quiescent.py` | 60 s no-leak streams (quiet, factory −28/−20/−12 dB, +10 dB level step): events/s, false alarms per hour, power. |
| `export_firmware.py` | Writes `firmware/wokwi_esp32s3/{snn_weights,test_streams}.h`. |
| `firmware/` | Event-driven SNN in C++ for the ESP32-S3, Wokwi circuit, host test. |
| `gpla12_experiment.py` | The same SNN recipe on real GPLA-12 recordings (12 classes). |

## Energy model (estimate, not measurement)
Arithmetic energy at 45 nm, 32-bit float (Horowitz 2014): mult 3.7 pJ, add 0.9 pJ.
* **Front-end** (common to all detectors): 256-pt FFT every 1 ms + band powers + log ≈ 2.6 k mult + 3.4 k add per frame ≈ **12.8 µW**.
* **SNN:** 1 accumulate per synaptic event. Neurons are updated either every step (clock-driven) or only when spikes arrive (event-driven, a lower bound).
* **CNN:** all MACs every frame. **Event-gated CNN:** CNN only while there was an input event in the last 100 ms, plus recomputing the receptive field on every wake-up.
* Not included: memory access, ADC, microphone, MCU static power. Real mW need a measurement on the board.

## Outputs (`runs/`)
`results.json` (per detector and FA budget: mean, std, per-seed values, CIs; `rq_quiescent_power_ratio`), `per_snr.csv`,
`det.csv`, `energy.csv`, `thresholds.json`, `quiescent.csv`, `figs/{snr_robustness,det,energy,event_raster,quiescent}.png`.
The files currently in `runs/` come from the first single-seed run (old evaluation). Rerun the commands above to replace them.
