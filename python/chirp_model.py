"""
FMCW ultrasonic radar - golden model, step 1.

Synthesize a linear FM chirp, place a target at a known range, mix the
echo against the transmit signal, and recover the beat frequency.

Run:   python python/chirp_model.py
Pass:  measured beat frequency lands within one FFT bin of the predicted value.

The whole point of this file is the PASS/FAIL line. It is not a demo -
it is the reference your Verilog will later be checked against, so it
has to be right before anything else gets built on top of it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------- waveform
C          = 343.0     # speed of sound in air, m/s
F_START    = 38e3      # chirp start frequency, Hz
BANDWIDTH  = 4e3       # sweep bandwidth, Hz  -> sweeps 38 kHz to 42 kHz
T_CHIRP    = 0.100     # chirp duration, s
FS         = 200e3     # simulation sample rate, Hz

# ------------------------------------------------------------------- scene
TARGETS = [(2.0, 0.5), (3.5, 0.3), (1.2, 0.4)]   # (range_m, amplitude)

# --------------------------------------------------------------- derived
SLOPE = BANDWIDTH / T_CHIRP        # Hz per second
N     = int(FS * T_CHIRP)
t     = np.arange(N) / FS


def chirp_phase(t):
    """Instantaneous phase of a linear FM sweep starting at F_START."""
    return 2 * np.pi * (F_START * t + 0.5 * SLOPE * t ** 2)


# ---------------------------------------------------------------- transmit
tx = np.cos(chirp_phase(t))

# ----------------------------------------------- receive (delayed echo)
rx = np.zeros(N)
for R, amp in TARGETS:
    tau_k = 2 * R / C
    echo = amp * np.cos(chirp_phase(t - tau_k))
    echo[t < tau_k] = 0.0
    rx += echo

# ------------------------------------------ mix (what the mixer does)
mixed = tx * rx                             # difference term + sum term

# ------------------------- low-pass: keep the beat, drop the sum term
sos  = butter(4, 8e3, btype="low", fs=FS, output="sos")
beat = sosfilt(sos, mixed)

# -------------------------------------------------------- range FFT
tau      = 2 * max(R for R, _ in TARGETS) / C   # slowest echo
valid    = beat[t >= tau]
window   = np.hanning(len(valid))
spectrum = np.abs(np.fft.rfft(valid * window))
freqs    = np.fft.rfftfreq(len(valid), 1 / FS)
bin_width = FS / len(valid)

print(f"FFT bin width : {bin_width:8.1f} Hz\n")

passed = True
for i, (R, amp) in enumerate(TARGETS):
    predicted = 2 * R * BANDWIDTH / (C * T_CHIRP)

    # look for the peak within +/- 5 bins of where theory says it is
    sel      = (freqs >= predicted - 5 * bin_width) & \
               (freqs <= predicted + 5 * bin_width)
    measured = freqs[sel][np.argmax(spectrum[sel])]

    est_range = measured * C * T_CHIRP / (2 * BANDWIDTH)
    ok = abs(measured - predicted) < bin_width
    passed &= ok

    print(f"target {i+1}:")
    print(f"  predicted beat : {predicted:8.1f} Hz")
    print(f"  measured  beat : {measured:8.1f} Hz")
    print(f"  est. range     : {est_range:7.3f} m   (true {R:.3f} m)")
    print(f"  {'ok' if ok else 'MISS'}\n")

print("PASS" if passed else "FAIL")

# -------------------------------------------------------------- plot
mask = freqs < 2000
plt.figure(figsize=(9, 4))
plt.plot(freqs[mask], spectrum[mask])
for R, _ in TARGETS:
    f_pred = 2 * R * BANDWIDTH / (C * T_CHIRP)
    plt.axvline(f_pred, color="r", ls="--", alpha=0.6,
                label=f"predicted {f_pred:.0f} Hz")
plt.xlabel("beat frequency (Hz)")
plt.ylabel("magnitude")
plt.title(f"Range FFT - {len(TARGETS)} targets")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()