import numpy as np
import Energy
import Utils
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def showPlots(early_mag_spectrum_log_smoothed, late_mag_spectrum_log_smoothed, frequencies, early_energy, late_energy, spectral_evolution_score):
    plt.figure()
    fig, axes = plt.subplots(1)
    plt.semilogx(frequencies, early_mag_spectrum_log_smoothed, label="Early", linestyle="--")
    plt.semilogx(frequencies, late_mag_spectrum_log_smoothed, label="Late", linestyle="-.")
    plt.legend()
    axes.set_xlabel("Frequency")
    fig.suptitle(f"Early = {np.round(early_energy, 2)}, Late = {np.round(late_energy, 2)}, Damping = {np.round(spectral_evolution_score, 2)} dB")
    plt.show()

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

    # Zero-pad to the same length
    pad_length = np.max([len(early_rir), len(late_rir)])
    early_rir = np.pad(early_rir, (0, pad_length - len(early_rir)), mode='constant')
    late_rir = np.pad(late_rir, (0, pad_length - len(late_rir)), mode='constant')

    # Get magnitude spectrum of each
    early_mag_spectrum = 20 * np.log10(np.abs(np.fft.rfft(early_rir)))
    late_mag_spectrum = 20 * np.log10(np.abs(np.fft.rfft(late_rir)))

    # Convert to log frequency from cutoff to Nyquist
    cutoff = 2000
    early_mag_spectrum_log, early_frequencies = Utils.linearToLog(early_mag_spectrum, sample_rate, cutoff, sample_rate / 2)
    late_mag_spectrum_log, late_frequencies = Utils.linearToLog(late_mag_spectrum, sample_rate, cutoff, sample_rate / 2)

    # Smooth spectra
    smoothing_window_length_samples = early_mag_spectrum_log.shape[0] // 2
    early_mag_spectrum_log_smoothed = savgol_filter(early_mag_spectrum_log, window_length=smoothing_window_length_samples, polyorder=1)
    late_mag_spectrum_log_smoothed = savgol_filter(late_mag_spectrum_log, window_length=smoothing_window_length_samples, polyorder=1)

    # Normalise both spectra so they overlap (compensate for the overall decay in level)
    early_mag_spectrum_log_smoothed -= np.max(early_mag_spectrum_log_smoothed)
    late_mag_spectrum_log_smoothed -= np.max(late_mag_spectrum_log_smoothed)

    # Get mean early and late magnitudes
    early_mean = np.mean(early_mag_spectrum_log_smoothed)
    late_mean = np.mean(late_mag_spectrum_log_smoothed)

    # Return (late - early) transformed to 0.2-0.8
    hf_damping_score = late_mean - early_mean
    hf_damping_score = (hf_damping_score + 24) / 28

    if should_show_plots:
        showPlots(early_mag_spectrum_log_smoothed, late_mag_spectrum_log_smoothed, early_frequencies, early_mean, late_mean, late_mean - early_mean)

    return hf_damping_score