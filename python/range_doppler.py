"""
FMCW ultrasonic radar - golden model, step 2: the range-Doppler map.

Builds a frame of N_CHIRPS sweeps, FFTs each row for range, then FFTs
down the columns for Doppler. The transpose between those two FFTs is
the corner turn - free here, a memory addressing problem in hardware.

Run:   python python/range_doppler.py
Pass:  both targets land in the predicted range and velocity cells.

NOTE ON COMPLEX MIXING
This model mixes to complex baseband (I and Q). That is what makes the
sign of the Doppler shift recoverable - approaching vs receding. A single
analog multiplier gives you a REAL beat signal only, and a real signal
cannot distinguish +v from -v: the spectrum is symmetric. To get the sign
in hardware you need two mixers driven 90 degrees apart. Worth knowing
before you build the front end.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------- waveform
C          = 343.0      # speed of sound in air, m/s
F_START    = 38e3       # chirp start frequency, Hz
BANDWIDTH  = 4e3        # sweep bandwidth, Hz   -> 38 kHz to 42 kHz
T_CHIRP    = 0.010      # chirp duration, s
FS         = 200e3      # sample rate, Hz
N_CHIRPS   = 64         # chirps per frame (the CPI)

# ------------------------------------------------------------------ scene
#            range_m,  velocity_m_s,  amplitude      (+v = approaching)
TARGETS = [(0.50,       0.035,        0.5),
           (0.80,      -0.015,        0.3)]

# ---------------------------------------------------------------- derived
SLOPE   = BANDWIDTH / T_CHIRP        # Hz per second
N       = int(FS * T_CHIRP)          # samples per chirp
t       = np.arange(N) / FS          # fast-time axis, one chirp
LAMBDA  = C / F_START                # wavelength that sets Doppler
CPI     = N_CHIRPS * T_CHIRP         # total observation time

# Every row of the frame must start at the same sample offset, so skip
# past the slowest echo. All targets are then present in every row.
MAX_TAU = 2 * max(r for r, _, _ in TARGETS) / C
SKIP    = int(MAX_TAU * FS)
N_VALID = N - SKIP


def chirp_phase(x):
    """Phase of the linear FM sweep. The x^2 term is the chirp."""
    return 2 * np.pi * (F_START * x + 0.5 * SLOPE * x ** 2)


# ------------------------------------------ build the frame, chirp by chirp
frame = np.zeros((N_CHIRPS, N_VALID), dtype=complex)

for k in range(N_CHIRPS):
    t_k  = k * T_CHIRP                        # elapsed time at this chirp
    beat = np.zeros(N, dtype=complex)

    for R0, v, amp in TARGETS:
        R_k   = R0 - v * t_k                  # target has moved
        tau_k = 2 * R_k / C
        # complex mix, tx * conj(rx), evaluated analytically
        phase = chirp_phase(t) - chirp_phase(t - tau_k)
        echo  = amp * np.exp(1j * phase)
        echo[t < tau_k] = 0.0                 # echo has not arrived yet
        beat += echo                          # air is linear: echoes add

    frame[k, :] = beat[SKIP:]                 # one row per chirp

NOISE = 1.0
frame += NOISE * (np.random.randn(N_CHIRPS, N_VALID) +
                  1j * np.random.randn(N_CHIRPS, N_VALID))

# ------------------------------------------------- 1) range FFT along rows
win_r     = np.hanning(N_VALID)
range_fft = np.fft.fft(frame * win_r[None, :], axis=1)

# --------------------------- 2) THE CORNER TURN: now read down the columns
#
# Rows went in one chirp at a time as data arrived. The Doppler FFT needs
# columns. In numpy, axis=0 is free. In your FPGA this is a BRAM written
# row-wise and read column-wise, and you cannot start until all N_CHIRPS
# rows have landed. That latency is the corner turn.
#
win_d  = np.hanning(N_CHIRPS)
rd_map = np.fft.fftshift(np.fft.fft(range_fft * win_d[:, None], axis=0),
                         axes=0)

magnitude = np.abs(rd_map)

# ------------------------------------------------------------------- axes
beat_freqs = np.fft.fftfreq(N_VALID, 1 / FS)
range_axis = beat_freqs * C * T_CHIRP / (2 * BANDWIDTH)

dopp_freqs = np.fft.fftshift(np.fft.fftfreq(N_CHIRPS, T_CHIRP))
vel_axis   = -dopp_freqs * LAMBDA / 2          # sign so +v = approaching

# keep the positive-range half, out to a sensible display limit
half   = N_VALID // 2
keep   = range_axis[:half] <= 1.5
r_ax   = range_axis[:half][keep]
mag    = magnitude[:, :half][:, keep]

# ------------------------------------------------------------ resolutions
dR      = C / (2 * BANDWIDTH)
dV      = LAMBDA / (2 * CPI)
v_unamb = LAMBDA / (4 * T_CHIRP)

print(f"range resolution     : {dR*100:6.2f} cm")
print(f"velocity resolution  : {dV*1000:6.2f} mm/s")
print(f"max unambiguous vel  : +/-{v_unamb:5.3f} m/s")
print(f"CPI                  : {CPI:6.3f} s")
print()

# ------------------------------------------ find the two strongest returns
work   = mag.copy()
passed = True

for i, (R0, v, _) in enumerate(TARGETS):
    idx      = np.unravel_index(np.argmax(work), work.shape)
    meas_v   = vel_axis[idx[0]]
    meas_r   = r_ax[idx[1]]

    ok_r = abs(meas_r - R0) < dR
    ok_v = abs(meas_v - v) < 3 * dV
    passed &= ok_r and ok_v

    print(f"peak {i+1}:")
    print(f"  range    measured {meas_r:6.3f} m     true {R0:6.3f} m"
          f"   {'ok' if ok_r else 'MISS'}")
    print(f"  velocity measured {meas_v:+6.3f} m/s   true {v:+6.3f} m/s"
          f"   {'ok' if ok_v else 'MISS'}")

    # blank this peak so the next argmax finds the other target
    r0 = max(0, idx[0] - 3); r1 = min(work.shape[0], idx[0] + 4)
    c0 = max(0, idx[1] - 3); c1 = min(work.shape[1], idx[1] + 4)
    work[r0:r1, c0:c1] = 0

print("\nPASS" if passed else "\nFAIL")

# ------------------------------------------------------------------- plot
mag_db = 20 * np.log10(mag / mag.max() + 1e-12)

plt.figure(figsize=(9, 5))
plt.pcolormesh(r_ax, vel_axis, mag_db, shading="auto",
               vmin=-40, vmax=0, cmap="viridis")

plt.colorbar(label="magnitude (dB)")

for R0, v, _ in TARGETS:
    plt.plot(R0, v, "rx", markersize=12, markeredgewidth=2)

plt.xlabel("range (m)")
plt.ylabel("velocity (m/s)   [+ = approaching]")
plt.title(f"Range-Doppler map  -  {N_CHIRPS} chirps, CPI {CPI:.2f} s")
plt.tight_layout()
plt.show()