import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt
from scipy.io import wavfile


def convolveWithProgItem(spatial_rir, prog_item_id):
    if prog_item_id == 1:
        filepath = "/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Programme Item Snippets/ClapShort.wav"
    elif prog_item_id == 2:
        filepath = "/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Programme Item Snippets/SaxShort.wav"
    else:
        assert False

    prog_item_snippet = wavfile.read(filepath)

    return np.convolve(spatial_rir, prog_item_snippet[1])


def findIndexOfClosest(list, target):
    list = np.asarray(list)
    index_of_closest = (np.abs(list - target)).argmin()
    return index_of_closest


def truncateSpectrum(spectrum, sample_rate, min_frequency, max_frequency):
    num_bins = len(spectrum)
    nyquist_frequency = sample_rate / 2.0
    bin_width = nyquist_frequency / num_bins
    min_frequency_index = int(np.floor(min_frequency / bin_width))
    max_frequency_index = int(np.floor(max_frequency / bin_width))
    truncated_spectrum = spectrum[min_frequency_index:max_frequency_index]
    new_freqs = np.linspace(min_frequency, max_frequency, len(truncated_spectrum))
    return truncated_spectrum, new_freqs


def interpolateList(list_to_interpolate, new_length):
    delta = (len(list_to_interpolate) - 1) / (new_length - 1)
    interpolated_list = np.zeros(new_length)

    for new_position in range(new_length):
        index_integer, index_fraction = int(new_position * delta // 1), new_position * delta % 1  # Split floating-point index into whole & fractional parts
        end_index = index_integer + 1 if index_fraction > 0 else index_integer  # Avoid index error
        interpolated_list[new_position] = ((1 - index_fraction) * list_to_interpolate[index_integer]
                                           + index_fraction * list_to_interpolate[end_index])

    return interpolated_list


def linearToLog(magnitudes, sample_rate, f_min, f_max):
    """
    Convert linear FFT magnitudes to logarithmic frequency space.

    Parameters:
        magnitudes (array): FFT magnitudes (assumes only positive frequencies from 0 Hz to Nyquist).
        sample_rate (float): Sampling rate in Hz.
        num_bins (int): Number of logarithmic frequency bins.
        f_min (float): Minimum frequency for log space.
        f_max (float): Maximum frequency, default is Nyquist.

    Returns:
        log_mags (array): Interpolated magnitudes in log frequency space between f_min and f_max
    """
    num_bins = len(magnitudes)
    nyquist = sample_rate / 2
    if f_max is None:
        f_max = nyquist

    # Linear frequency axis (positive frequencies only)
    lin_freqs_full = np.linspace(0, nyquist, num_bins)

    # Remove DC component (0 Hz) to avoid log(0)
    lin_freqs_full = lin_freqs_full[1:]
    magnitudes = magnitudes[1:]

    # Logarithmic frequency axis
    log_freqs_trunc = np.logspace(np.log10(f_min), np.log10(f_max), num_bins)

    # Interpolation
    interp = interp1d(lin_freqs_full, magnitudes, kind='linear', bounds_error=False, fill_value=0.0)
    log_mags = interp(log_freqs_trunc)

    return log_mags, log_freqs_trunc


def getMidpointsBetween(values):
    return [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]


def getThirdOctaveBandCentres(min_frequency, max_frequency):
    reference_frequency = 1000.0
    increase_factor = 2 ** (1 / 3)

    # Find lower and upper band indices relative to 1000 Hz
    min_band_index = int(np.ceil(np.log10(min_frequency / reference_frequency) / np.log10(increase_factor)))
    max_band_index = int(np.floor(np.log10(max_frequency / reference_frequency) / np.log10(increase_factor)))

    # Calculate centre frequencies
    centre_frequencies = [reference_frequency * (increase_factor ** n) for n in range(min_band_index, max_band_index + 1)]

    return centre_frequencies


def getMeanMagnitudeBetweenFrequencies(half_spectrum_magnitudes, sample_rate, min_frequency, max_frequency):
    magnitudes_between = truncateSpectrum(half_spectrum_magnitudes, sample_rate, min_frequency, max_frequency)
    mean = np.mean(magnitudes_between)
    return mean


def getOctaveBandsFromIR(rir, sample_rate, octave_band_resolution=1):
        # Octave bands
        octave_band_centres = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
        num_bands = len(octave_band_centres)
        filter_order = 5
        centre_to_crossover_factor = 2 ** (1 / 2)

        band_signals = np.zeros([len(rir), num_bands])
        for freq_idx, centre_freq in enumerate(octave_band_centres):
            band_type = ('low' if freq_idx == 0
                         else ('high' if freq_idx == num_bands - 1
                               else 'band'))

            bin_lower = centre_freq / centre_to_crossover_factor
            bin_upper = centre_freq * centre_to_crossover_factor

            if band_type == 'low':
                sos = butter(2 * filter_order, bin_upper, 'lowpass', fs=sample_rate, output='sos')
            elif band_type == 'band':
                sos = butter(filter_order, (bin_lower, bin_upper), 'bandpass', fs=sample_rate, output='sos')
            elif band_type == 'high':
                sos = butter(2 * filter_order, bin_lower, 'highpass', fs=sample_rate, output='sos')
            else:
                raise ValueError('"band_type" must be one of ["low", "band", "high"].')

            band_signals[:, freq_idx] = sosfilt(sos, rir)

        # Returns (bands x samples)
        return band_signals, octave_band_centres


# cartesian_coords: shape = [N, 3 (x, y, z)]
def cartesianToSpherical(cartesian_coords):
    spherical_coords = np.zeros_like(cartesian_coords)
    squared_lateral_plane = cartesian_coords[:, 0] ** 2 + cartesian_coords[:, 1] ** 2
    spherical_coords[:, 0] = np.sqrt(squared_lateral_plane + cartesian_coords[:, 2] ** 2)
    spherical_coords[:, 1] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])
    spherical_coords[:, 2] = np.arctan2(cartesian_coords[:, 2], np.sqrt(squared_lateral_plane)) # for elevation angle defined from Z-axis down
    return spherical_coords


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def getFrequencyIndexRange(frequency_indices, min_frequency, max_frequency, sample_rate):
    min_freq_index = int(np.floor(len(frequency_indices) * (min_frequency / (sample_rate / 2))))
    max_freq_index = int(np.floor(len(frequency_indices) * (max_frequency / (sample_rate / 2))))
    frequency_index_range = range(min_freq_index, max_freq_index)

    return frequency_index_range


def applyEqualLoudnessContour(mag_spectrum_dB, mag_spectrum_freqs):
    # ISO 226:2003 standard frequencies
    equal_loudness_freqs = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250,
        1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500
    ])

    # Approximate equal-loudness corrections in dB at 60 phons
    equal_loudness_magnitudes_dB = np.array([
        78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1,
        17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5,
        1.7, 1.3, 4.2, 6.0, 5.4, 1.5, 6.0, 12.6, 13.9, 12.3
    ])

    # Interpolate equal-loudness curve
    equal_loudness_curve = interp1d(equal_loudness_freqs, equal_loudness_magnitudes_dB, kind="linear", bounds_error=False, fill_value="extrapolate")

    # Get correction for given frequencies
    correction_curve = equal_loudness_curve(mag_spectrum_freqs)

    # Apply correction
    corrected_magnitudes_dB = mag_spectrum_dB - correction_curve

    return corrected_magnitudes_dB


def circularStd(radii, angles_rad):
    # Weighted circular statistics
    x = np.sum(radii * np.cos(angles_rad))
    y = np.sum(radii * np.sin(angles_rad))
    R = np.sqrt(x ** 2 + y ** 2) / np.sum(radii)

    # Circular standard deviation
    return np.sqrt(-2 * np.log(R))