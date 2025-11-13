import numpy as np
from scipy.signal import savgol_filter

def getEDC(rir, sample_rate):
    integration_limit_samples = len(rir)
    reversed_rir = rir[::-1]

    # Calculate Schroeder decay
    edc_dB_reversed = 10.0 * np.log10(np.cumsum(np.square(reversed_rir)) / np.sum(np.square(rir)))
    edc_dB = edc_dB_reversed[::-1]

    time_values_samples = range(integration_limit_samples)
    time_values_seconds = [time_value / sample_rate for time_value in time_values_samples]

    return edc_dB, time_values_seconds


# ETC using rectangular window with 0% overlap
def getEnergyTimeCurve(rir, sample_rate, window_duration_ms: float = 10.0):
    rir /= np.max(np.abs(rir))
    window_length_samples = int((sample_rate * window_duration_ms) / 1000)
    num_rir_samples = len(rir)
    squared_rir = np.square(rir)

    energy_time_curve = np.zeros(int(num_rir_samples / window_length_samples))

    for window_index, sample_index in enumerate(range(0, int(num_rir_samples - window_length_samples), window_length_samples)):
        summation = np.sum(squared_rir[sample_index:sample_index + window_length_samples])
        energy_time_curve[window_index] = 10 * np.log10(summation)

    time_values = [(energy_bin * window_length_samples) / sample_rate for energy_bin in range(len(energy_time_curve))]

    return energy_time_curve, time_values


def getEnergySpectrum(rir, sample_rate, fft_size, etc_window_duration_ms):
        # Compute energy time curve
        etc, etc_times = getEnergyTimeCurve(rir, sample_rate, etc_window_duration_ms)

        # Set smoothing window length such that the minimum energy frequency doesn't get smoothed over
        min_frequency_to_preserve = 1 # Hz
        min_period = 1 / min_frequency_to_preserve
        smoothing_window_length_samples = int(np.floor(min_period / (etc_window_duration_ms / 1000)))

        # Mirror the start and ends of the ETC before smoothing (avoids edge effects), then clip ends after smoothing
        # window = np.hanning(smoothing_window_length_samples)
        etc_mirror_start = etc[smoothing_window_length_samples:0:-1]
        etc_mirror_end = etc[-1:-smoothing_window_length_samples - 1:-1]
        etc_mirror_padded = np.concat([etc_mirror_start, etc, etc_mirror_end])
        # smoothed_etc_padded = np.convolve(window, etc_mirror_padded, 'same')
        smoothed_etc_padded = savgol_filter(etc_mirror_padded, window_length=smoothing_window_length_samples, polyorder=2)
        smoothed_etc = smoothed_etc_padded[smoothing_window_length_samples:-smoothing_window_length_samples]

        # Divide ETC by smoothed to remove decay shape
        etc_over_smoothed = etc / smoothed_etc

        # Subtract mean to centre about 0
        etc_over_smoothed_sub_mean = etc_over_smoothed - np.mean(etc_over_smoothed)

        # Get magnitude of energy spectrum
        energy_spectrum = np.fft.rfft(etc_over_smoothed_sub_mean, n=fft_size)

        return abs(energy_spectrum)