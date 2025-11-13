import Utils
import numpy as np
from matplotlib import pyplot as plt
import Energy
from scipy.signal import butter, sosfilt


def showEnergySpectrumPlots(energy_spectrum_dB, energy_spectrum_freqs, flutter_score):
    plt.plot(energy_spectrum_freqs, energy_spectrum_dB)
    plt.title(f"|FFT(Energy Decay Fluctuations)| (flutter = {round(flutter_score, 3)})")

    plt.show()

def showACFPlots(num_octave_bands, auto_correlations, sample_rate, octave_band_centres, flutter_score, etc_window_duration_ms):
    fig, axes = plt.subplots(num_octave_bands)
    fig.set_size_inches(6, 8)
    fig.set_layout_engine("tight")
    plt.suptitle(f"Auto-Correlation Function of Energy Decay (flutter = {round(flutter_score, 3)})")

    times = np.arange(0, auto_correlations.shape[0]) * (etc_window_duration_ms / 1000)
    # frequencies = 1 / (np.clip(bin_indices, 0.00001, None) )

    for octave_band in range(num_octave_bands):
        auto_correlation = auto_correlations[:, octave_band]
        # axes[octave_band].plot(times, 20 * np.log10(np.clip(auto_correlation, 0.00001, 1)))
        axes[octave_band].plot(times, auto_correlation)
        axes[octave_band].set_title(f"{octave_band_centres[octave_band]} Hz")
        axes[octave_band].set_xlim([0.05, 0.5])
        # axes[octave_band].set_ylim([-200, 400])

    plt.show()

def getScoreSingleChannel(rir, sample_rate, should_show_plots=False):
    # High-pass RIR from 1 kHz
    filter_order = 4
    cutoff_Hz = 2000.0
    sos = butter(filter_order, cutoff_Hz, 'highpass', fs=sample_rate, output='sos')
    rir_high_passed = sosfilt(sos, rir)

    # Get energy time curve of the high-passed RIR
    etc_window_duration_ms = 2.0
    etc_dB, _ = Energy.getEnergyTimeCurve(rir_high_passed, sample_rate, etc_window_duration_ms)

    # Truncate after -40 dB
    etc_dB_trunc = etc_dB[:Utils.findIndexOfClosest(etc_dB, -40.0)]

    # Get energy spectrum (FFT of energy time curve in decibels)
    fft_size = 2 ** 10
    energy_spectrum_dB = np.log10(np.abs(np.fft.rfft(etc_dB_trunc, n=fft_size)))

    # Truncate energy spectrum between 0-30 Hz
    energy_spectrum_freqs = np.fft.rfftfreq(fft_size, etc_window_duration_ms / 1000.0)
    energy_frequency_index_range = Utils.getFrequencyIndexRange(energy_spectrum_freqs,
                                                                0.0,
                                                                20.0,
                                                                sample_rate=1.0 / (etc_window_duration_ms / 1000.0))
    energy_spectrum_dB = energy_spectrum_dB[energy_frequency_index_range]

    # Find max magnitude of energy oscillations between 0-20 Hz minus the mean and standard deviation
    flutter_echo_score = 1.0 - (np.max(energy_spectrum_dB) - np.mean(energy_spectrum_dB) - np.std(energy_spectrum_dB))

    if should_show_plots:
        showEnergySpectrumPlots(energy_spectrum_dB,
                                energy_spectrum_freqs[energy_frequency_index_range],
                                flutter_echo_score)

    return flutter_echo_score


def getFlutterEchoScore(spatial_rir, sample_rate, should_show_plots=False):
    # Compute flutter score for the omnidirectional and interaural bidirectional channels
    scores = [getScoreSingleChannel(spatial_rir[:, channel], sample_rate, should_show_plots) for channel in range(2)]

    # Output transformed summation of the channel scores
    return (np.sum(scores) - 0.9) * 1.7