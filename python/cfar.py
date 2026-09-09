"""
FMCW ultrasonic radar - golden model, step 3: CA-CFAR detection.

Replaces argmax with a real detector. For every cell, estimate the local
noise from a ring of training cells, then threshold relative to that
estimate. The threshold moves with the noise floor, which is the whole
point: you specify a false alarm RATE, not a level.

Run:   python python/cfar.py
Pass:  all targets detected, and the measured false alarm rate on
       noise-only data matches the designed P_fa within a factor of ~2.

WHY GUARD CELLS
A real target occupies several cells - the FFT main lobe is a few bins
wide. If those neighbours end up in the training set, the target inflates
its own threshold and masks itself. Guard cells are the buffer that keeps
target energy out of the noise estimate.

WHY POWER, NOT MAGNITUDE
After the FFTs, complex Gaussian noise gives Rayleigh-distributed
magnitude and exponentially-distributed POWER. The alpha formula below
assumes exponential, so CFAR runs on |x|^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# --------------------------------------------------------------- waveform
C          = 343.0
F_START    = 38e3
BANDWIDTH  = 4e3
T_CHIRP    = 0.040
FS         = 200e3
N_CHIRPS   = 32

# ------------------------------------------------------------------ scene
#            range_m,  velocity_m_s,  amplitude
TARGETS = [(0.60,      0.030,        1.00),
           (1.40,     -0.020,        0.50),
           (2.20,      0.010,        0.25)]

NOISE      = 1.0        # complex noise sigma per sample
MAX_RANGE  = 3.0        # how far out to process

# ------------------------------------------------------------- CFAR setup
PFA        = 1e-3       # designed false alarm probability per cell
GUARD_1D   = 2
TRAIN_1D   = 6
GUARD_2D   = 2          # half-width, so guard box is 5x5
TRAIN_2D   = 1          # half-width, so full box is 13x13

# ---------------------------------------------------------------- derived
SLOPE   = BANDWIDTH / T_CHIRP
N       = int(FS * T_CHIRP)
t       = np.arange(N) / FS
LAMBDA  = C / F_START
CPI     = N_CHIRPS * T_CHIRP
MAX_TAU = 2 * max(r for r, _, _ in TARGETS) / C
SKIP    = int(MAX_TAU * FS)
N_VALID = N - SKIP


def chirp_phase(x):
    return 2 * np.pi * (F_START * x + 0.5 * SLOPE * x ** 2)


def make_rd_map(targets, noise_sigma, seed=None):
    """Build one range-Doppler frame. Returns power map and axes."""
    rng   = np.random.default_rng(seed)
    frame = np.zeros((N_CHIRPS, N_VALID), dtype=complex)

    for k in range(N_CHIRPS):
        t_k  = k * T_CHIRP
        beat = np.zeros(N, dtype=complex)
        for R0, v, amp in targets:
            R_k   = R0 - v * t_k
            tau_k = 2 * R_k / C
            echo  = amp * np.exp(1j * (chirp_phase(t) - chirp_phase(t - tau_k)))
            echo[t < tau_k] = 0.0
            beat += echo
        frame[k, :] = beat[SKIP:]

    frame += noise_sigma * (rng.standard_normal((N_CHIRPS, N_VALID)) +
                            1j * rng.standard_normal((N_CHIRPS, N_VALID)))

    rng_fft = np.fft.fft(frame * np.hanning(N_VALID)[None, :], axis=1)
    rd      = np.fft.fftshift(
                  np.fft.fft(rng_fft * np.hanning(N_CHIRPS)[:, None], axis=0),
                  axes=0)

    beat_f     = np.fft.fftfreq(N_VALID, 1 / FS)
    range_axis = beat_f * C * T_CHIRP / (2 * BANDWIDTH)
    dopp_f     = np.fft.fftshift(np.fft.fftfreq(N_CHIRPS, T_CHIRP))
    vel_axis   = -dopp_f * LAMBDA / 2

    half = N_VALID // 2
    keep = range_axis[:half] <= MAX_RANGE
    power = np.abs(rd[:, :half][:, keep]) ** 2      # POWER, not magnitude

    return power, range_axis[:half][keep], vel_axis


# ------------------------------------------------------------------ alpha
def cfar_alpha(n_train_cells, pfa):
    """
    Scale factor for cell-averaging CFAR on exponentially distributed
    power. Derivation: with N training cells the noise estimate Z is the
    sample mean, and P_fa = (1 + alpha/N)^-N. Solve for alpha.

    Note the cost of a small training set: fewer cells means a noisier
    estimate, so alpha has to be larger to hold P_fa - which costs you
    detection sensitivity. That is the CFAR loss.
    """
    return n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0)


# -------------------------------------------------------------- 1D CA-CFAR
def ca_cfar_1d(power, n_guard, n_train, pfa):
    """One range profile in, threshold and detection mask out."""
    kernel = np.ones(2 * (n_guard + n_train) + 1)
    kernel[n_train:n_train + 2 * n_guard + 1] = 0.0   # blank CUT + guards
    n_cells = int(kernel.sum())
    kernel /= n_cells                                 # make it an average

    noise_est = np.convolve(power, kernel, mode="same")
    alpha     = cfar_alpha(n_cells, pfa)
    threshold = alpha * noise_est

    valid = np.zeros_like(power, dtype=bool)
    edge  = n_guard + n_train
    valid[edge:-edge] = True                          # edges lack neighbours

    return threshold, (power > threshold) & valid, alpha


# -------------------------------------------------------------- 2D CA-CFAR
def ca_cfar_2d(power, n_guard, n_train, pfa):
    """Full range-Doppler map in, threshold and detection mask out."""
    w_full  = 2 * (n_guard + n_train) + 1
    w_guard = 2 * n_guard + 1

    # sum over the full box minus sum over the guard box = training sum
    sum_full  = uniform_filter(power, size=w_full,  mode="nearest") * w_full ** 2
    sum_guard = uniform_filter(power, size=w_guard, mode="nearest") * w_guard ** 2

    n_cells   = w_full ** 2 - w_guard ** 2
    noise_est = (sum_full - sum_guard) / n_cells

    alpha     = cfar_alpha(n_cells, pfa)
    threshold = alpha * noise_est

    valid = np.zeros_like(power, dtype=bool)
    e     = n_guard + n_train
    valid[e:-e, e:-e] = True

    return threshold, (power > threshold) & valid, alpha


# ============================================================ run it
power, r_ax, v_ax = make_rd_map(TARGETS, NOISE, seed=1)

print(f"map size        : {power.shape[0]} Doppler x {power.shape[1]} range")
print(f"range bin       : {r_ax[1]-r_ax[0]:.4f} m")
print(f"designed P_fa   : {PFA:.1e}\n")

# ---- 1D: one range profile, taken at the strongest target's velocity ----
row_idx  = np.argmin(np.abs(v_ax - TARGETS[0][1]))
profile  = power[row_idx, :]
thr1, det1, a1 = ca_cfar_1d(profile, GUARD_1D, TRAIN_1D, PFA)

print(f"1D: {2*TRAIN_1D} training cells, alpha = {a1:.2f} "
      f"({10*np.log10(a1):.1f} dB above local noise)")
print(f"    detections at ranges: "
      f"{', '.join(f'{r:.2f}' for r in r_ax[det1])} m\n")

# ---- 2D: the whole map ----
thr2, det2, a2 = ca_cfar_2d(power, GUARD_2D, TRAIN_2D, PFA)
n_train_2d = (2*(GUARD_2D+TRAIN_2D)+1)**2 - (2*GUARD_2D+1)**2

print(f"2D: {n_train_2d} training cells, alpha = {a2:.2f} "
      f"({10*np.log10(a2):.1f} dB)")

hits = np.argwhere(det2)
print(f"    {len(hits)} cells above threshold")

found = []
for R0, v0, _ in TARGETS:
    near = [(v_ax[i], r_ax[j]) for i, j in hits
            if abs(r_ax[j] - R0) < 0.15 and abs(v_ax[i] - v0) < 0.01]
    found.append(len(near) > 0)
    print(f"    target at {R0:.2f} m, {v0:+.3f} m/s -> "
          f"{'DETECTED' if near else 'MISSED'}")

# ---- false alarm rate: same processing, no targets, many trials ----
print("\nfalse alarm check (noise only, 20 frames)...")
fa_total = valid_total = 0
for s in range(20):
    p_noise, _, _ = make_rd_map([(2.20, 0.0, 0.0)], NOISE, seed=100 + s)
    _, d, _ = ca_cfar_2d(p_noise, GUARD_2D, TRAIN_2D, PFA)
    e = GUARD_2D + TRAIN_2D
    fa_total    += d.sum()
    valid_total += d[e:-e, e:-e].size

measured_pfa = fa_total / valid_total
print(f"  designed P_fa : {PFA:.2e}")
print(f"  measured P_fa : {measured_pfa:.2e}   "
      f"({fa_total} alarms in {valid_total} cells)")

pfa_ok = 0.3 * PFA < measured_pfa < 3 * PFA
print("\nPASS" if all(found) and pfa_ok else "\nFAIL")

# ================================================================== plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

db = lambda x: 10 * np.log10(x + 1e-30)

ax1.plot(r_ax, db(profile), label="range profile")
ax1.plot(r_ax, db(thr1), "r--", label="CFAR threshold")
ax1.plot(r_ax[det1], db(profile[det1]), "go", markersize=9,
         fillstyle="none", markeredgewidth=2, label="detections")
ax1.set_xlabel("range (m)")
ax1.set_ylabel("power (dB)")
ax1.set_title(f"1D CA-CFAR on one Doppler row (v = {v_ax[row_idx]:+.3f} m/s)")
ax1.legend()
ax1.grid(alpha=0.3)

m = db(power)
ax2.pcolormesh(r_ax, v_ax, m, shading="auto",
               vmin=m.max() - 50, vmax=m.max(), cmap="viridis")
if len(hits):
    ax2.plot(r_ax[hits[:, 1]], v_ax[hits[:, 0]], "r.", markersize=4,
             label="CFAR detections")
for R0, v0, _ in TARGETS:
    ax2.plot(R0, v0, "wx", markersize=12, markeredgewidth=2)
ax2.set_xlabel("range (m)")
ax2.set_ylabel("velocity (m/s)")
ax2.set_title("2D CA-CFAR - white x = truth, red dots = detections")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()