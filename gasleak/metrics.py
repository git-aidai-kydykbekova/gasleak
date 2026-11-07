"""
Alarm logic and detection metrics (shared by evaluate.py and quiescent.py).

An alarm is raised at the first step t where the detector output stays above a threshold
for K consecutive steps.  Thresholds are in logit space (sigmoid outputs saturate at 1.0).

Leak-free ("negative") segments: every no-leak clip after the encoder warm-up AND the part of
every leak clip before the onset.  An alarm inside any of them is a false alarm.
"""
import numpy as np

WARMUP = 100        # ms used by the encoder to learn the noise floor
K = 5               # consecutive ms above threshold to declare a detection


def window_min(z):
    """z: [N, T] -> q[:, t] = min(z[:, t-K+1 .. t]);  q > tau  <=>  K consecutive steps above tau."""
    q = np.full(z.shape, -np.inf, np.float32)
    q[:, K - 1:] = np.lib.stride_tricks.sliding_window_view(z, K, axis=1).min(-1)
    return q


def _valid(T, lo, hi):
    """[N, T] mask of steps whose whole K-window lies inside [lo, hi)."""
    t = np.arange(T)
    return (t >= np.asarray(lo)[:, None] + K - 1) & (t < np.asarray(hi)[:, None])


def first_alarm(q, tau, lo, hi):
    """first step with q > tau inside [lo, hi); -1 if none."""
    ok = (q > tau) & _valid(q.shape[1], lo, hi)
    return np.where(ok.any(1), ok.argmax(1), -1)


def peak(q, lo, hi):
    """max windowed score inside [lo, hi): an alarm happens in the segment iff tau < peak."""
    return np.where(_valid(q.shape[1], lo, hi), q, -np.inf).max(1)


def onsets_ms(meta, T):
    """leak onset in ms (T for no-leak clips, i.e. the whole clip is leak-free)."""
    return np.where(meta.label.values == 1, (meta.onset_s.fillna(0).values * 1000).round(), T).astype(int)


def peaks(q, meta):
    """(peak in leak-free segment, peak after onset) per clip; post-onset peak is -inf for no-leak clips."""
    T = q.shape[1]
    on = onsets_ms(meta, T)
    neg = peak(q, np.full(len(on), WARMUP), on)
    pos = np.where(meta.label.values == 1, peak(q, on, np.full(len(on), T)), -np.inf)
    return neg, pos


def pick_threshold(q, meta, target_fa):
    """lowest threshold whose false-alarm rate over leak-free segments is <= target_fa (on this set)."""
    neg, _ = peaks(q, meta)
    return float(np.quantile(neg, 1 - target_fa, method="higher"))


def det_curve(q, meta):
    """(false-alarm rate, miss rate) for every threshold: the full detection-error trade-off."""
    neg, pos = peaks(q, meta)
    pos = pos[meta.label.values == 1]
    taus = np.unique(np.concatenate([neg, pos]))
    taus = taus[np.isfinite(taus)]
    fa = (neg[None] > taus[:, None]).mean(1)
    miss = (pos[None] <= taus[:, None]).mean(1)
    return fa, miss


def score(q, meta, tau):
    """per-clip outcomes at threshold tau."""
    T = q.shape[1]
    lab = meta.label.values
    on = onsets_ms(meta, T)
    neg_alarm = first_alarm(q, tau, np.full(len(on), WARMUP), on) >= 0     # alarm while no leak present
    det = first_alarm(q, tau, on, np.full(len(on), T))
    tp = (lab == 1) & (det >= 0)
    delay = np.where(tp, det - on, np.nan)
    return dict(tp=tp, neg_alarm=neg_alarm, delay=delay)


def summarize(r, meta, T):
    lab = meta.label.values
    pos = lab == 1
    tp, fa_clip, fa_pre = r["tp"].sum(), r["neg_alarm"][~pos].sum(), r["neg_alarm"][pos].sum()
    on = onsets_ms(meta, T)
    neg_hours = (on - WARMUP).sum() / 3.6e6
    rec = tp / max(pos.sum(), 1)
    prec = tp / max(tp + fa_clip + fa_pre, 1)
    d = r["delay"][r["tp"]]
    nan = float("nan")
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                false_alarm_rate=r["neg_alarm"].mean(),             # over all leak-free segments
                false_alarm_rate_noleak_clips=fa_clip / max((~pos).sum(), 1),
                pre_onset_alarm_rate=fa_pre / max(pos.sum(), 1),
                alarmed_segments_per_hour=(fa_clip + fa_pre) / neg_hours,
                median_delay_ms=float(np.median(d)) if len(d) else nan,
                mean_delay_ms=float(np.mean(d)) if len(d) else nan,
                p90_delay_ms=float(np.percentile(d, 90)) if len(d) else nan)


def alarm_events(q, tau, start=WARMUP):
    """number of separate alarms (rising edges of q > tau) per recording, for long streams."""
    a = q[:, start:] > tau
    return (a[:, 1:] & ~a[:, :-1]).sum(1) + a[:, 0]
