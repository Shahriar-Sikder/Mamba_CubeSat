"""Synthetic LEO CubeSat telemetry, anomaly injection and preprocessing.

Pipeline
--------
generate_telemetry()  -> raw signals (with NaNs), orbital phase, eclipse flag, labels
preprocess()          -> temporal split, z-score (nominal train only), forward-fill + mask,
                         orbital encoding  -> model-ready arrays per split
make_windows()        -> sliding windows of length `window` (optionally nominal-only)

Conventions
-----------
* Time step dt = 10 s. One orbit = 95 min = 570 samples, eclipse = 35 min = 210 samples.
* `miss` mask: 1 = value was MISSING (forward-filled), 0 = observed.
* `labels`: 1 = anomalous timestep, 0 = nominal.
* `event_id`: index into `events` list for anomalous timesteps, -1 otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGNAL_NAMES = [
    "batt_voltage",   # EPS  [V]
    "solar_current",  # EPS  [A], exactly 0 in eclipse
    "temperature",    # thermal [degC], lagged sinusoid
    "omega_x",        # ADCS [deg/s]
    "omega_y",        # ADCS [deg/s]
    "omega_z",        # ADCS [deg/s]
    "rssi",           # comms [dBm], with dropouts (NaN)
]
ORBIT_FEATURES = ["sin_nu", "cos_nu", "eclipse"]


@dataclass
class TelemetryConfig:
    dt_s: float = 10.0
    period_min: float = 95.0
    eclipse_min: float = 35.0
    n_orbits: int = 72
    seed: int = 0
    # anomaly counts
    n_spikes: int = 40
    n_contextual: int = 16
    # fraction of point/contextual events placed in train / val / test. Fewer in train
    # keeps more nominal training windows; more in test gives stable event-wise metrics.
    event_split_probs: tuple = (0.2, 0.3, 0.5)
    fade_orbits: float = 4.0          # collective battery fade over the last N orbits
    fade_final_capacity: float = 0.45  # capacity fraction reached at end of fade
    # missing data
    p_point_missing: float = 0.002     # isolated missing samples on every channel
    n_comm_dropouts: int = 60          # RSSI dropout segments
    split_fracs: tuple = (0.6, 0.2, 0.2)

    @property
    def samples_per_orbit(self) -> int:
        return int(round(self.period_min * 60 / self.dt_s))

    @property
    def eclipse_samples(self) -> int:
        return int(round(self.eclipse_min * 60 / self.dt_s))

    @property
    def n_samples(self) -> int:
        return self.n_orbits * self.samples_per_orbit


# --------------------------------------------------------------------------------------
# 1. Generation
# --------------------------------------------------------------------------------------
def orbital_phase(cfg: TelemetryConfig):
    """True anomaly nu(t) in [0, 2pi) for a circular orbit and the eclipse flag.

    Eclipse is centred on nu = pi (anti-sun side) and lasts `eclipse_min`.
    """
    n = cfg.n_samples
    t = np.arange(n) * cfg.dt_s
    nu = (2 * np.pi * t / (cfg.period_min * 60)) % (2 * np.pi)
    half_ecl = np.pi * cfg.eclipse_min / cfg.period_min  # half-width in radians
    eclipse = (np.abs(nu - np.pi) < half_ecl).astype(np.float32)
    return t, nu, eclipse


def _simulate_eps(nu, eclipse, rng, capacity):
    """Solar current + battery voltage from a simple state-of-charge model.

    capacity: array (N,) of relative capacity (1.0 = healthy). Capacity fade makes the
    eclipse discharge deeper and raises internal resistance.
    """
    n = len(nu)
    sunlit = 1.0 - eclipse
    # Solar current: sun-angle shaped + tumbling ripple, exactly 0 in eclipse.
    shape = 0.80 + 0.20 * np.cos(nu) + 0.05 * np.sin(6 * nu)
    solar_i = sunlit * (1.15 * shape + rng.normal(0, 0.015, n))
    solar_i = np.clip(solar_i, 0.0, None) * sunlit

    load = 0.55 + 0.03 * np.sin(3 * nu)       # platform load [A]
    q_nom = 0.55 * 10.0 * 1100.0              # nominal capacity [A*s], ~0.2 SoC per eclipse
    soc = np.empty(n)
    s = 0.95
    eta = 0.9
    for i in range(n):
        charge = eta * solar_i[i]
        if s > 0.9:                           # CC/CV taper near full charge
            charge *= max(0.0, (1.0 - s) / 0.1)
        s = np.clip(s + (charge - load[i]) * 10.0 / (q_nom * capacity[i]), 0.0, 1.0)
        soc[i] = s
    net = eta * solar_i - load
    r_int = 0.10 / capacity                    # internal resistance grows with fade
    v = 7.0 + 1.2 * soc + r_int * net + rng.normal(0, 0.004, n)
    return v, solar_i, soc


def generate_telemetry(cfg: TelemetryConfig | None = None) -> dict:
    """Generate multivariate telemetry with labelled anomalies.

    Returns a dict with keys:
      t, nu, eclipse, signals (N, 7, NaN = missing), clean (N, 7, before anomalies/NaNs),
      labels (N,), event_id (N,), events (list of dicts), names, cfg
    """
    cfg = cfg or TelemetryConfig()
    rng = np.random.default_rng(cfg.seed)
    t, nu, eclipse = orbital_phase(cfg)
    n, spo = cfg.n_samples, cfg.samples_per_orbit

    # --- collective anomaly: slow battery capacity fade at the end of the mission ----
    capacity = np.ones(n)
    fade_start = n - int(cfg.fade_orbits * spo)
    ramp = np.linspace(1.0, cfg.fade_final_capacity, n - fade_start)
    capacity[fade_start:] = ramp

    v, solar_i, _ = _simulate_eps(nu, eclipse, rng, capacity)

    # Thermal: lagged sinusoid (peak ~0.6 rad after sub-solar point) + slow drift + noise
    temp = 8.0 + 14.0 * np.cos(nu - 0.6) + 0.5 * np.sin(2 * np.pi * t / (24 * 3600)) \
        + rng.normal(0, 0.25, n)

    # ADCS: three noisy angular rates (B-dot / magnetorquer residual at 2x orbital rate)
    omega = np.stack([
        0.10 * np.sin(2 * nu + 0.3),
        0.08 * np.cos(2 * nu + 1.1),
        0.05 * np.sin(nu + 2.0),
    ], axis=1)
    omega += np.cumsum(rng.normal(0, 0.0005, (n, 3)), axis=0) * 0.1  # slow bias walk
    omega += rng.normal(0, 0.02, (n, 3))

    # Comms: RSSI with orbital variation; dropouts become NaN later
    rssi = -88.0 + 5.0 * np.sin(2 * nu + 0.4) + rng.normal(0, 0.8, n)

    clean = np.column_stack([v, solar_i, temp, omega, rssi]).astype(np.float64)
    signals = clean.copy()
    labels = np.zeros(n, dtype=np.int64)
    event_id = -np.ones(n, dtype=np.int64)
    events: list[dict] = []

    def add_event(kind, start, end, channel):
        eid = len(events)
        events.append(dict(id=eid, kind=kind, start=int(start), end=int(end), channel=channel))
        labels[start:end] = 1
        event_id[start:end] = eid

    def free(start, end, margin=20):
        return labels[max(0, start - margin):end + margin].sum() == 0

    # split boundaries (same as temporal_split) used to place events
    a = int(n * cfg.split_fracs[0])
    b = int(n * (cfg.split_fracs[0] + cfg.split_fracs[1]))
    ranges = [(50, a - 50), (a + 50, b - 50), (b + 50, fade_start - 50)]

    def random_time():
        lo, hi = ranges[rng.choice(3, p=cfg.event_split_probs)]
        return int(rng.integers(lo, hi))

    # --- point anomalies: 1-2 sample spikes on a random channel ----------------------
    chan_std = np.nanstd(clean, axis=0)
    k = 0
    while k < cfg.n_spikes:
        s = random_time()
        length = int(rng.integers(1, 3))
        c = int(rng.integers(0, clean.shape[1]))
        if not free(s, s + length):
            continue
        sign = rng.choice([-1.0, 1.0])
        signals[s:s + length, c] += sign * rng.uniform(5, 9) * chan_std[c]
        add_event("point", s, s + length, SIGNAL_NAMES[c])
        k += 1

    # --- contextual anomalies: a NORMAL value at the WRONG orbital phase --------------
    #  (a) solar current at a typical sunlit level during eclipse (stuck sensor / mis-ID)
    #  (b) temperature held at its warm sunlit value during eclipse (heater stuck on)
    ecl_starts = np.where(np.diff(eclipse) > 0)[0] + 1
    k = 0
    while k < cfg.n_contextual:
        e0 = int(ecl_starts[np.argmin(np.abs(ecl_starts - random_time()))])
        # the whole eclipse must lie inside one split range (and before the fade)
        if not any(lo <= e0 and e0 + cfg.eclipse_samples <= hi for lo, hi in ranges):
            continue
        s = e0 + int(rng.integers(30, cfg.eclipse_samples - 60))
        length = int(rng.integers(20, 45))
        if not free(s, s + length):
            continue
        if k % 2 == 0:
            ref = solar_i[(eclipse == 0)]
            signals[s:s + length, 1] = rng.choice(ref, length)  # values from sunlit dist.
            add_event("contextual", s, s + length, "solar_current")
        else:
            warm = np.percentile(temp, 85)
            signals[s:s + length, 2] = warm + rng.normal(0, 0.25, length)
            add_event("contextual", s, s + length, "temperature")
        k += 1

    add_event("collective", fade_start, n, "batt_voltage")

    # --- missing data (not anomalies) ------------------------------------------------
    miss = rng.random(signals.shape) < cfg.p_point_missing
    for _ in range(cfg.n_comm_dropouts):
        s = int(rng.integers(0, n - 40))
        miss[s:s + int(rng.integers(5, 35)), 6] = True
    signals[miss] = np.nan

    return dict(t=t, nu=nu, eclipse=eclipse, signals=signals, clean=clean,
                capacity=capacity, labels=labels, event_id=event_id, events=events,
                names=list(SIGNAL_NAMES), cfg=cfg)


# --------------------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------------------
def temporal_split(n: int, fracs=(0.6, 0.2, 0.2)) -> dict[str, slice]:
    """Contiguous, non-shuffled split: train | val | test (no look-ahead leakage)."""
    assert abs(sum(fracs) - 1.0) < 1e-9
    a = int(n * fracs[0])
    b = int(n * (fracs[0] + fracs[1]))
    return {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}


class ZScore:
    """Per-channel z-score. Fit ONLY on nominal, observed training samples."""

    def fit(self, x: np.ndarray, nominal: np.ndarray):
        xs = x[nominal.astype(bool)]
        self.mean = np.nanmean(xs, axis=0)
        self.std = np.nanstd(xs, axis=0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


def ffill_with_mask(x: np.ndarray):
    """Forward-fill NaNs along time. Leading NaNs -> 0 (= train mean after z-score).

    Returns (filled, miss) where miss[t, c] = 1 if x[t, c] was NaN.
    """
    miss = np.isnan(x)
    filled = x.copy()
    for c in range(x.shape[1]):
        col = filled[:, c]
        idx = np.where(~miss[:, c], np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        col[np.isnan(col)] = 0.0                 # leading NaNs
    return filled, miss.astype(np.float32)


def orbital_encoding(nu, eclipse):
    return np.column_stack([np.sin(nu), np.cos(nu), eclipse]).astype(np.float32)


def preprocess(raw: dict, fracs=None) -> dict:
    """Split -> z-score (nominal train) -> ffill + mask -> concat orbital features.

    Each split dict contains:
      X  (T, 2C+3) model input  = [z-scored signals | miss mask | sin nu, cos nu, eclipse]
      Y  (T, C)    targets      = z-scored, forward-filled signals
      M  (T, C)    miss mask    (1 = missing; excluded from loss / score)
      labels (T,), event_id (T,), t (T,), eclipse (T,)
    """
    cfg = raw["cfg"]
    fracs = fracs or cfg.split_fracs
    sig = raw["signals"]
    n, c = sig.shape
    splits = temporal_split(n, fracs)

    tr = splits["train"]
    scaler = ZScore().fit(sig[tr], nominal=(raw["labels"][tr] == 0))

    out = {"scaler": scaler, "splits": splits, "names": raw["names"],
           "input_names": raw["names"] + [f"miss_{s}" for s in raw["names"]] + ORBIT_FEATURES}
    for name, sl in splits.items():
        z = scaler.transform(sig[sl])
        zf, miss = ffill_with_mask(z)            # ffill per split -> no cross-split leakage
        orb = orbital_encoding(raw["nu"][sl], raw["eclipse"][sl])
        X = np.concatenate([zf.astype(np.float32), miss, orb], axis=1)
        out[name] = dict(X=X, Y=zf.astype(np.float32), M=miss,
                         labels=raw["labels"][sl], event_id=raw["event_id"][sl],
                         t=raw["t"][sl], eclipse=raw["eclipse"][sl])
        assert X.shape == (sl.stop - sl.start, 2 * c + 3), X.shape
        assert not np.isnan(X).any(), f"NaNs left in {name}"
    return out


def make_windows(split: dict, window: int = 512, stride: int = 64, nominal_only: bool = True):
    """Sliding windows over one split.

    Returns dict of arrays: X (W, L, Cin), Y (W, L, C), M (W, L, C), labels (W, L), start (W,)
    If nominal_only, windows containing ANY labelled anomaly are dropped.
    """
    T = len(split["X"])
    starts = np.arange(0, T - window + 1, stride)
    if nominal_only:
        starts = np.array([s for s in starts if split["labels"][s:s + window].sum() == 0],
                          dtype=np.int64)
    idx = starts[:, None] + np.arange(window)[None, :]
    return dict(X=split["X"][idx], Y=split["Y"][idx], M=split["M"][idx],
                labels=split["labels"][idx], start=starts)


def build_dataset(cfg: TelemetryConfig | None = None, window=512, train_stride=64, val_stride=128):
    """One-call convenience used by the training notebook."""
    raw = generate_telemetry(cfg)
    proc = preprocess(raw)
    proc["train_windows"] = make_windows(proc["train"], window, train_stride, nominal_only=True)
    proc["val_windows"] = make_windows(proc["val"], window, val_stride, nominal_only=True)
    return raw, proc


# --------------------------------------------------------------------------------------
# 3. Plotting
# --------------------------------------------------------------------------------------
SERIES_COLOR = "#2a78d6"
ANOM_COLOR = "#d03b3b"
ECL_COLOR = "#8a8a85"


def shade_spans(ax, mask, color, alpha, label=None, x0=0):
    """Shade runs where mask is True. `x0` = x-coordinate of mask[0]."""
    mask = np.asarray(mask).astype(bool)
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(x0 + s - 0.5, x0 + e - 0.5, color=color, alpha=alpha, lw=0,
                   label=label if i == 0 else None)


def plot_telemetry(raw: dict, start: int, n_orbits: float = 3, channels=None, title=None):
    """Small multiples (one axis per channel), eclipse in grey, anomalies in red."""
    import matplotlib.pyplot as plt

    spo = raw["cfg"].samples_per_orbit
    end = min(start + int(n_orbits * spo), len(raw["t"]))
    channels = channels if channels is not None else list(range(len(raw["names"])))
    sl = slice(start, end)
    x = np.arange(start, end)
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 1.6 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, channels):
        ax.plot(x, raw["signals"][sl, c], color=SERIES_COLOR, lw=1.0)
        shade_spans(ax, raw["eclipse"][sl], ECL_COLOR, 0.15, "eclipse", x0=start)
        shade_spans(ax, raw["labels"][sl], ANOM_COLOR, 0.35, "anomaly", x0=start)
        ax.set_ylabel(raw["names"][c], rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=2)
    axes[-1].set_xlabel("sample index (10 s / sample)")
    fig.suptitle(title or f"Telemetry, samples {start}-{end}", fontsize=11)
    fig.tight_layout()
    return fig
