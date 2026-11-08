"""
Synthetic ultrasonic gas-leak dataset generator.

Physics (simplified):
  * A pressurised gas leak is a turbulent jet -> stationary broadband hiss whose
    energy peaks in the ultrasonic range (~25-50 kHz), with a fast onset when the
    leak starts.
  * Industrial background noise is mostly < 20 kHz (machines, motors, impacts),
    with a weaker ultrasonic tail, plus short ultrasonic transients
    (pneumatic exhaust bursts, valve clicks) and occasional ultrasonic tones
    (switch-mode power supplies) that act as "hard negatives".

Each clip: 1.0 s at 96 kHz.  SNR is defined in the sensor's ultrasonic analysis
band (18-46 kHz): SNR = 10 log10(P_leak / P_background) inside that band.
"""
import numpy as np
from scipy import signal

FS = 96_000
DUR = 1.0
N = int(FS * DUR)
BAND = (18_000, 46_000)
_SOS_BAND = signal.butter(4, BAND, btype="bandpass", fs=FS, output="sos")


def band_power(x):
    return np.mean(signal.sosfilt(_SOS_BAND, x) ** 2) + 1e-20


def pink_noise(rng, n):
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    X = np.fft.rfft(w) / np.sqrt(f)
    return np.fft.irfft(X, n)


def _norm(x):
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def background(rng, kind, n=N):
    """kind: 'quiet' | 'factory'.  Returns background with unit-ish band power scale applied later."""
    t = np.arange(n) / FS
    x = np.zeros(n)
    # broadband pink noise (falls off -> little ultrasonic energy)
    x += _norm(pink_noise(rng, n)) * (0.3 if kind == "quiet" else 1.0)
    if kind == "factory":
        # rotating machinery: fundamental + harmonics, gear-mesh tone
        f0 = rng.uniform(20, 120)
        for h in range(1, 12):
            x += 0.4 / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
        fg = rng.uniform(1_000, 8_000)
        x += 0.3 * np.sin(2 * np.pi * fg * t) * (1 + 0.3 * np.sin(2 * np.pi * f0 * t))
        # impacts / clanks: decaying resonances
        for _ in range(rng.poisson(3 * n / N)):         # ~3 impacts per second
            t0 = rng.integers(0, n - 4800)
            fr = rng.uniform(2_000, 25_000)
            L = rng.integers(1000, 4800)
            tt = np.arange(L) / FS
            x[t0:t0 + L] += rng.uniform(0.5, 3) * np.exp(-tt * rng.uniform(80, 400)) * np.sin(2 * np.pi * fr * tt)
    # broadband ultrasonic floor (electronics / air flow), small
    x += 0.02 * rng.standard_normal(n)
    return x


def ultrasonic_distractors(rng, prob_burst, prob_tone, n=N):
    """Hard negatives in the ultrasonic band. Returned un-scaled.
    Bursts: per second of signal, with probability prob_burst, 1-3 bursts.  Tone: whole signal."""
    d = np.zeros(n)
    t = np.arange(n) / FS
    for _ in range(max(1, round(n / N))):
        if rng.random() >= prob_burst:       # pneumatic exhaust / valve hiss bursts (short!)
            continue
        for _ in range(rng.integers(1, 4)):
            L = int(rng.uniform(0.01, 0.08) * FS)
            t0 = rng.integers(0, n - L)
            lo = rng.uniform(18_000, 30_000)
            sos = signal.butter(2, (lo, min(lo + rng.uniform(8_000, 20_000), 47_000)), "bandpass", fs=FS, output="sos")
            env = np.hanning(L) ** 0.5
            d[t0:t0 + L] += signal.sosfilt(sos, rng.standard_normal(L)) * env * rng.uniform(1, 4)
    if rng.random() < prob_tone:             # SMPS / ultrasonic tone, stationary but narrowband
        fo = rng.uniform(20_000, 45_000)
        d += 0.5 * np.sin(2 * np.pi * fo * t)
    return d


def leak(rng, onset_s, pressure):
    """Turbulent-jet hiss. Higher pressure -> higher peak frequency. Un-scaled."""
    fp = np.interp(pressure, [0, 1], [24_000, 40_000]) * rng.uniform(0.9, 1.1)
    bw = fp * rng.uniform(0.5, 0.9)
    lo, hi = max(fp - bw / 2, 15_000), min(fp + bw / 2, 47_000)
    sos = signal.butter(2, (lo, hi), "bandpass", fs=FS, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(N))
    t = np.arange(N) / FS
    am = 1 + rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * rng.uniform(2, 15) * t)   # jet flutter
    rise = rng.uniform(0.003, 0.02)
    env = np.clip((t - onset_s) / rise, 0, 1)
    return x * am * env


# absolute ultrasonic-band background levels (dB re arbitrary sensor unit)
LEVELS = {"quiet": -40.0, "factory": (-28.0, -12.0)}


def make_clip(rng, *, label, scene, snr_db=None, pressure=None, onset_s=None, n=N, level_db=None):
    """n: length in samples (long no-leak recordings: n > N).  level_db: fixed background level
    in the ultrasonic band instead of the scene's default / random range."""
    assert label == 0 or n == N, "leak clips are 1 s long"
    bg = background(rng, scene, n)
    bg += ultrasonic_distractors(rng, prob_burst=0.15 if scene == "quiet" else 0.6,
                                 prob_tone=0.05 if scene == "quiet" else 0.25, n=n)
    # scale background to target absolute ultrasonic level
    if level_db is not None:
        lvl = level_db
    else:
        lvl = LEVELS[scene] if scene == "quiet" else rng.uniform(*LEVELS[scene])
    bg *= np.sqrt(10 ** (lvl / 10) / band_power(bg))
    x = bg.copy()
    if label == 1:
        lk = leak(rng, onset_s, pressure)
        active = slice(int(onset_s * FS), N)
        p_lk = band_power(lk[active.start:]) if N - active.start > 2000 else band_power(lk)
        lk *= np.sqrt(band_power(bg) * 10 ** (snr_db / 10) / p_lk)
        x += lk
    return x.astype(np.float32)


def sample_meta(rng, label):
    scene = "quiet" if rng.random() < 0.4 else "factory"
    m = dict(label=label, scene=scene, snr_db=np.nan, pressure=np.nan, onset_s=np.nan)
    if label == 1:
        m.update(snr_db=float(rng.choice([-10, -5, 0, 5, 10, 20])),
                 pressure=float(rng.uniform(0, 1)),
                 # leak starts inside the clip so detection delay can be measured
                 onset_s=float(rng.uniform(0.15, 0.6)))
    return m
