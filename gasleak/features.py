"""
Front-end: ultrasonic filterbank -> band levels (dB) -> event (spike) encoding.

Event encoder (2 spike channels per band, inspired by cochlear sustained/onset cells):
  * noise-floor tracker per band (slow EMA, frozen while signal is above floor)
  * excess  e = max(0, L_dB - floor_dB - margin)
  * SUSTAINED channel: integrate-and-fire on e  -> spike rate ~ how far above floor
  * ONSET channel:     delta modulator on e (spike when e rises by >= theta_on)
Stationary background (any loudness) -> e ~ 0 -> almost no events (quiescent).
"""
import numpy as np
from scipy import signal
from .synth import FS

HOP = 96          # 1 ms
NFFT = 256
N_BANDS = 16
F_LO, F_HI = 18_000, 46_000


def _band_matrix():
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    edges = np.geomspace(F_LO, F_HI, N_BANDS + 1)
    M = np.zeros((N_BANDS, len(f)), np.float32)
    for b in range(N_BANDS):
        M[b, (f >= edges[b]) & (f < edges[b + 1])] = 1
    M /= M.sum(1, keepdims=True)
    return M, edges


BAND_M, BAND_EDGES = _band_matrix()


def band_levels_db(x, tau_ms=4.0):
    """x: 1-D waveform -> [T, N_BANDS] smoothed band level in dB (1 ms frames)."""
    _, _, Z = signal.stft(x, fs=FS, window="hann", nperseg=NFFT, noverlap=NFFT - HOP,
                          boundary=None, padded=False)
    P = (np.abs(Z) ** 2).astype(np.float32)            # [F, T]
    E = BAND_M @ P                                     # [B, T]
    a = np.exp(-1.0 / tau_ms)
    E = signal.lfilter([1 - a], [1, -a], E, axis=1)    # EMA smoothing
    return (10 * np.log10(E + 1e-14)).T.astype(np.float32)   # [T, B]


def encode_events(L, init_frames=100, margin_db=3.0, tau_floor_s=2.0,
                  theta_sus=6.0, theta_on=1.5):
    """L: [N_clips, T, B] dB levels -> spikes uint8 [N_clips, T, 2B] (sustained | onset)."""
    n, T, B = L.shape
    floor = np.median(L[:, :init_frames], axis=1)              # init from pre-roll
    a = 1.0 - 1.0 / (tau_floor_s * 1000)
    acc = np.zeros((n, B), np.float32)
    ref = np.zeros((n, B), np.float32)
    out = np.zeros((n, T, 2 * B), np.uint8)
    for t in range(T):
        lt = L[:, t]
        quiet = lt < floor + margin_db
        floor = np.where(quiet, a * floor + (1 - a) * lt, floor)   # frozen when active
        e = np.maximum(0.0, lt - floor - margin_db)
        # sustained: integrate-and-fire
        acc += e
        s = acc >= theta_sus
        acc -= s * theta_sus
        # onset: delta modulation (ON only), reference relaxes back down
        on = (e - ref) >= theta_on
        ref = np.where(on, ref + theta_on, np.minimum(ref, e))
        out[:, t, :B] = s
        out[:, t, B:] = on
    return out
