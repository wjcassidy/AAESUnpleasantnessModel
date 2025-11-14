import numpy as np
import Energy
import Utils
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def drawPlot(mag_spectrum_log_smoothed, frequencies, mean):
    plt.semilogx(frequencies, mag_spectrum_log_smoothed, label=f"Mean Magnitude = {np.round(mean, 2)} dB")
    plt.legend()


def getEarlyAndLateRIR(rir, sample_rate, early_start_dB, early_end_dB, late_start_dB, late_end_dB):
    edc_dB, _ = Energy.getEDC(rir, sample_rate)
    early_start_samples = Utils.findIndexOfClosest(edc_dB, early_start_dB)
    early_end_samples = Utils.findIndexOfClosest(edc_dB, early_end_dB)
    late_start_samples = Utils.findIndexOfClosest(edc_dB, late_start_dB)
    late_end_samples = Utils.findIndexOfClosest(edc_dB, late_end_dB)

    early_rir = rir[early_start_samples:early_end_samples]
    late_rir = rir[late_start_samples:late_end_samples]

    return early_rir, late_rir


def getHFDampingScore(rir, sample_rate, should_show_plots=False):
    # Split early and late regions of the RIR
    early_rir, late_rir = getEarlyAndLateRIR(rir, sample_rate, -1, -15, -35, -40)

    # indexed by early/late region
    mean_magnitudes = np.zeros(2)

    if should_show_plots:
        plt.figure()

    for rir_index, rir_region in enumerate([early_rir, late_rir]):
        # Get magnitude spectrum of each
        mag_spectrum = 20 * np.log10(np.abs(np.fft.rfft(rir_region)))

        # Convert to log frequency from cutoff to Nyquist
        cutoff = 2000
        mag_spectrum_log, frequencies = Utils.linearToLog(mag_spectrum, sample_rate, cutoff, sample_rate / 2)

        # Smooth spectra
        smoothing_window_length_samples = mag_spectrum_log.shape[0] // 2
        mag_spectrum_log_smoothed = savgol_filter(mag_spectrum_log, window_length=smoothing_window_length_samples, polyorder=1)

        # Normalise both spectra so they overlap (compensate for the overall decay in level)
        mag_spectrum_log_smoothed -= np.max(mag_spectrum_log_smoothed)

        # Get mean early and late magnitudes
        mean_magnitudes[rir_index] = np.mean(mag_spectrum_log_smoothed)

        if should_show_plots:
            drawPlot(mag_spectrum_log_smoothed, frequencies, mean_magnitudes[rir_index])

    # Return (late - early) transformed to 0.2-0.8
    hf_damping_score = mean_magnitudes[1] - mean_magnitudes[0]
    hf_damping_score = (hf_damping_score + 28) / 33

    if should_show_plots:
        plt.title(f"HF Damping Score = {np.round(hf_damping_score, 2)}")
        plt.show()

    return hf_damping_score