import numpy as np
import matplotlib.pyplot as plt
import RT
import Utils
import Energy
from scipy.signal import savgol_filter


def showPlots(mag_minus_mean_dB, colouration_score, mag_spectrum_log_trunc, mag_spectrum_smoothed, mag_over_means, mag_spectrum_freqs):
    plt.figure()
    fig, axes = plt.subplots(3)
    fig.set_size_inches(5, 5)
    fig.set_layout_engine("tight")
    plt.suptitle(f"Colouration (stddev of bottom plot) = {round(colouration_score, 3)}")
    axes[0].set_xscale("log")
    axes[0].set_title('Mag Raw (dashed) and Smoothed (solid)')
    axes[0].plot(mag_spectrum_freqs, mag_spectrum_log_trunc, 'c')
    axes[0].plot(mag_spectrum_freqs, mag_spectrum_smoothed, 'black',)
    axes[0].set_xticks([20, 200, 2000])
    axes[0].set_xticklabels(["20", "200", "2k"])
    axes[1].set_xscale("log")
    axes[1].set_title('Mag Minus Smoothed')
    axes[1].plot(mag_spectrum_freqs, mag_minus_mean_dB, 'c')
    axes[1].set_xticks([20, 200, 2000])
    axes[1].set_xticklabels(["20", "200", "2k"])
    axes[2].set_xscale("log")
    axes[2].set_title('Mag Minus Smoothed Linear (Equal Loudness)')
    axes[2].plot(mag_spectrum_freqs, mag_over_means, 'black')
    axes[2].set_xticks([20, 200, 2000])
    axes[2].set_xticklabels(["20", "200", "2k"])
    plt.show()


def getColouration(rir, sample_rate, should_show_plots=False):
    rir_num_samples = len(rir)

    # Estimate T30 from -5 dB to -35 dB
    rt = RT.estimateRT(rir, sample_rate, start_dB=-5, end_dB=-35)

    # Window the RIR between the 0 dB and -40 dB times
    edc_dB, time_values = Energy.getEDC(rir, sample_rate)
    trunc_start_samples = Utils.findIndexOfClosest(edc_dB, 0)
    trunc_end_samples = Utils.findIndexOfClosest(edc_dB, -40)

    rir_sample_indices = range(rir_num_samples)
    rir_sample_indices_windowed = rir_sample_indices[trunc_start_samples:trunc_end_samples]

    # Compensate for IR decay shape (multiply IR by exp(6.91 * t / RT))
    sampling_period = 1.0 / sample_rate
    rir_windowed_compensated = [rir[sample_index] * np.exp(6.91 * sample_index * sampling_period / rt)
                                for sample_index in rir_sample_indices_windowed]

    # Get magnitude spectrum
    fft_size = 2 ** 17
    mag_spectrum = np.abs(np.fft.rfft(rir_windowed_compensated, fft_size))

    # Truncate result (Schroeder frequency lower, 2 kHz upper) and convert spectrum to log frequency
    room_volume = 5000 # assumed
    schroeder_frequency = 2000.0 * np.sqrt(rt / room_volume)
    lower_frequency_limit = schroeder_frequency
    upper_frequency_limit = 8000 # modified from 4 kHz

    mag_spectrum_log_trunc_linear, mag_spectrum_freqs = Utils.linearToLog(mag_spectrum, sample_rate, lower_frequency_limit, upper_frequency_limit)

    # Convert magnitude to decibels (modification)
    mag_spectrum_log_trunc_dB = 20 * np.log10(mag_spectrum_log_trunc_linear)

    # Get smoothed spectrum, mirroring start and ends for one window length to avoid edge effects
    num_octaves = np.log10(mag_spectrum_freqs[-1] / mag_spectrum_freqs[0]) / np.log10(2)
    window_size = int((len(mag_spectrum_log_trunc_dB) / num_octaves) * 0.15) # Smooth in 0.15 * octave bands
    mirrored_bins_start = mag_spectrum_log_trunc_dB[window_size:0:-1]
    mirrored_bins_end = mag_spectrum_log_trunc_dB[:-window_size - 1:-1]

    mag_spectrum_to_smooth = np.concat([mirrored_bins_start, mag_spectrum_log_trunc_dB, mirrored_bins_end])
    mag_spectrum_smoothed = savgol_filter(mag_spectrum_to_smooth, window_size, 1)
    mag_spectrum_smoothed = mag_spectrum_smoothed[window_size:-window_size]

    # Subtract smoothed magnitude from raw (modification)
    mag_minus_mean_dB = mag_spectrum_log_trunc_dB - mag_spectrum_smoothed

    # Apply equal-loudness contour
    mag_minus_mean_equal_loud_dB = Utils.applyEqualLoudnessContour(mag_minus_mean_dB, mag_spectrum_freqs)
    mag_minus_mean_equal_loud_linear = 10 ** (mag_minus_mean_equal_loud_dB / 20)

    # Output standard deviation * peakedness (modification)
    std_dev_linear = np.std(mag_minus_mean_equal_loud_linear)
    peakedness = np.max(mag_minus_mean_dB)
    colouration_score = std_dev_linear * peakedness

    # Scale to approximately 0-1 (modification)
    colouration_score = (colouration_score - 2.5) / 14

    if should_show_plots:
        showPlots(mag_minus_mean_dB,
                  colouration_score,
                  mag_spectrum_log_trunc_dB,
                  mag_spectrum_smoothed,
                  mag_minus_mean_equal_loud_linear,
                  mag_spectrum_freqs)

    return colouration_score
