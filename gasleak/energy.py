"""
Compute-energy model (an estimate, not a measurement).

Per-operation energy, 32-bit float, 45 nm CMOS (Horowitz, ISSCC 2014):
    mult 3.7 pJ, add / compare 0.9 pJ  ->  MAC 4.6 pJ, accumulate (AC) 0.9 pJ.
Arithmetic only: memory access, ADC, microphone and MCU static/leakage power are NOT included.

Every pipeline = front-end (filterbank, common to all) + encoder + network:
  * front-end : Hann window, 256-pt real FFT every 1 ms, |Z|^2 of the 18-46 kHz bins, band sums,
                EMA smoothing, log10                                   (gasleak/features.py)
  * encoder   : noise-floor tracker + sustained/onset spike generation (SNN, gated CNN)
                or floor subtraction / scaling of band levels           (CNN)
  * network   : SNN  - one AC per synaptic event + neuron state updates
                CNN  - MACs of every conv layer, every frame
                gated CNN - CNN only while woken up by input events
"""
import numpy as np
from .features import NFFT, N_BANDS, BAND_M

E_MUL, E_ADD = 3.7e-12, 0.9e-12
E_MAC, E_AC = E_MUL + E_ADD, E_ADD
FPS = 1000                                   # frames (time steps) per second
NEURON_UPDATE = E_MUL + 3 * E_ADD            # m = beta*m + I + b, threshold compare, reset
LOG_MACS = 8                                 # log10 via table + polynomial, MAC-equivalents
HOLD_MS = 100                                # gated CNN stays awake this long after the last event


def frontend_ops():
    """(mults, adds) per 1 ms frame for the shared filterbank front-end."""
    m = NFFT // 2                                         # real FFT via an N/2-point complex FFT
    butterflies = (m // 2) * int(np.log2(m)) + m // 2     # radix-2 stages + real-FFT split step
    n_bins = int((BAND_M > 0).any(0).sum())               # only 18-46 kHz bins are used
    mul = NFFT + 4 * butterflies + 2 * n_bins + N_BANDS + 2 * N_BANDS + LOG_MACS * N_BANDS
    add = 6 * butterflies + n_bins + n_bins + N_BANDS + LOG_MACS * N_BANDS
    return mul, add


def frontend_power_uW():
    mul, add = frontend_ops()
    return (mul * E_MUL + add * E_ADD) * FPS * 1e6


def encoder_power_uW(kind):
    """event encoder: floor tracker (2 mul, 2 add/cmp), excess (2 add), integrate-and-fire (3 add),
    delta modulator (3 add) per band.  CNN input: floor subtraction + affine scaling per band."""
    if kind == "events":
        return N_BANDS * (2 * E_MUL + 10 * E_ADD) * FPS * 1e6
    return N_BANDS * (E_ADD + E_MAC) * FPS * 1e6


def snn_energy(model, ev_count, s1_count, s2_count):
    """Per-step network energy (J) of LeakSNN, [N, T] each, from spike counts per step.
    clock : every neuron updated every step (conservative, straightforward MCU loop).
    event : a layer's neurons are updated only on steps where that layer receives or emits spikes
            (lazy leak: decay over the gap via a beta^dt lookup) - a lower bound."""
    n_hid = model.fc1.out_features
    syn =(ev_count * n_hid + s1_count * n_hid + s2_count * 1) * E_AC
    clock = syn + (2 * n_hid + 1) * NEURON_UPDATE
    act1, act2, act3 = (ev_count + s1_count) > 0, (s1_count + s2_count) > 0, s2_count > 0
    event = syn + (n_hid * act1 + n_hid * act2 + act3) * NEURON_UPDATE
    return clock, event


def dense_twin_power_uW(model):
    """the same SNN evaluated as a dense frame-based network (all weights used every step)."""
    n_in, n_hid = model.fc1.in_features, model.fc1.out_features
    return ((n_in * n_hid + n_hid * n_hid + n_hid) * E_MAC + (2 * n_hid + 1) * NEURON_UPDATE) * FPS * 1e6


def cnn_step_energy(model):
    """J per output step: conv MACs + bias adds + ReLU compares."""
    n_out = sum(c.out_channels for c in model.convs)
    return model.macs_per_step() * E_MAC + (2 * n_out + 1) * E_ADD


def cnn_receptive_field(model):
    return 1 + (model.k - 1) * sum(model.dil)


def wake_mask(ev_count, hold=HOLD_MS):
    """gated CNN: awake at t iff there was at least one input event in (t-hold, t]."""
    c = np.cumsum(ev_count > 0, axis=1)
    past = np.concatenate([np.zeros((c.shape[0], hold)), c[:, :-hold]], 1)
    return (c - past) > 0


def gated_cnn_energy(model, awake):
    """Per-step CNN energy while awake, plus re-computing the receptive field on every wake-up
    (intermediate activations are stale after sleeping; band levels are kept in a ring buffer)."""
    wakeups = awake & ~np.concatenate([np.zeros((awake.shape[0], 1), bool), awake[:, :-1]], 1)
    return cnn_step_energy(model) * (awake + wakeups * (cnn_receptive_field(model) - 1))


PIPELINES = ("snn (clock-driven)", "snn (event-driven)", "dense twin", "cnn (always on)", "cnn (event-gated)")


def pipeline_power(snn, cnn, ev_count, s1_count, s2_count, sel):
    """Mean power (µW) of every pipeline over the time steps selected by sel [N, T] (bool).
    ev_count, s1_count, s2_count: [N, T] spikes per step (encoder output, SNN hidden layers 1, 2).
    Returns ({pipeline: {frontend_uW, encoder_uW, network_uW, total_uW}}, extras)."""
    def to_uW(e):
        return float(e[sel].mean() * FPS * 1e6)
    clock, event = snn_energy(snn, ev_count, s1_count, s2_count)
    awake = wake_mask(ev_count)
    enc_ev, enc_dense = encoder_power_uW("events"), encoder_power_uW("dense")
    parts = {"snn (clock-driven)": (enc_ev, to_uW(clock)),
             "snn (event-driven)": (enc_ev, to_uW(event)),
             "dense twin": (enc_ev, dense_twin_power_uW(snn)),
             "cnn (always on)": (enc_dense, cnn_step_energy(cnn) * FPS * 1e6),
             "cnn (event-gated)": (enc_ev + enc_dense, to_uW(gated_cnn_energy(cnn, awake)))}
    fe = frontend_power_uW()
    out = {k: dict(frontend_uW=fe, encoder_uW=e, network_uW=n, total_uW=fe + e + n) for k, (e, n) in parts.items()}
    return out, dict(input_events_per_s=float(ev_count[sel].mean() * FPS), gated_cnn_duty=float(awake[sel].mean()))
