import warnings

import numpy as np
from matplotlib import pyplot as plt
import Utils
import Energy


# spatial_ir: impulse response in B-format
def getDOAPerSample(spatial_ir, window_length_samples=5):
    # Get each axis as the product of the omni channel and each respective bidirectional channel, smoothed with hanning
    assert (window_length_samples >= 5)
    window = np.hanning(window_length_samples) # Note: this is slightly different to the MATLAB Hanning window
    coords_cartesian = np.zeros([spatial_ir.shape[0], 3])

    for axis in range(3):
        spherical_harmonic_index = axis + 1
        coords_cartesian[:, axis] = np.convolve(window, spatial_ir[:, 0] * spatial_ir[:, spherical_harmonic_index], "same")

    # Normalise each direction to a radius of 1
    euclidean_distances = np.tile(np.sqrt(np.square(coords_cartesian[:, 0])
                                          + np.square(coords_cartesian[:, 1])
                                          + np.square(coords_cartesian[:, 2])), (3, 1)).transpose()

    doa_per_sample_cartesian = coords_cartesian / euclidean_distances

    return doa_per_sample_cartesian


# Returns plot_angles_rad, radii_dB
def getSpatioTemporalMap(spatial_ir,
                         sample_rate,
                         start_ms=-1,
                         duration_ms=200,
                         start_is_relative_to_direct=True,
                         plane="transverse",
                         num_plot_angles=300):
    doa_cartesian = getDOAPerSample(spatial_ir)

    start_samples = int(np.floor(sample_rate * start_ms / 1000))
    duration_samples = int(np.floor(sample_rate * duration_ms / 1000))

    if start_is_relative_to_direct:
        # Assumes the direct sound arrives as the maximum sample of the omni channel
        direct_sample_index = np.argmax(np.abs(spatial_ir[:, 0]))

        # Set start relative to the direct sample
        start_index = np.max([0, direct_sample_index + start_samples])
        end_index = np.min([spatial_ir.shape[0] - 1, start_index + duration_samples])
    else:
        start_index = start_samples
        end_index = start_index + duration_samples

    # Truncate DOAs to time region
    doa_cartesian_trunc = doa_cartesian[start_index:end_index, :]

    # Transform cartesian coords and convert to spherical, where doa_spherical = (radii, azimuths, elevations)
    # Note: doa_spherical[:, 0] will be ignored as radius is taken from pressure
    if plane == "lateral":
        doa_spherical_rad = Utils.cartesianToSpherical(doa_cartesian_trunc)
    elif plane == "median":
        transformed_doa_cartesian = np.zeros_like(doa_cartesian_trunc)
        transformed_doa_cartesian[:, 0] = doa_cartesian_trunc[:, 0]
        transformed_doa_cartesian[:, 1] = doa_cartesian_trunc[:, 2]
        transformed_doa_cartesian[:, 2] = -doa_cartesian_trunc[:, 1]
        doa_spherical_rad = Utils.cartesianToSpherical(transformed_doa_cartesian)
    elif plane == "transverse":
        transformed_doa_cartesian = np.zeros_like(doa_cartesian_trunc)
        transformed_doa_cartesian[:, 0] = doa_cartesian_trunc[:, 2]
        transformed_doa_cartesian[:, 1] = doa_cartesian_trunc[:, 1]
        transformed_doa_cartesian[:, 2] = -doa_cartesian_trunc[:, 0]
        doa_spherical_rad = Utils.cartesianToSpherical(transformed_doa_cartesian)
        doa_spherical_rad[:, 1] += np.pi / 2
    else:
        warnings.warn("Plane argument not recognised (defaulting to 'lateral')")
        doa_spherical_rad = Utils.cartesianToSpherical(doa_cartesian_trunc)

    # Apply arbitrary offsets for alignment correction
    azimuth_offset_rad = np.pi
    elevation_offset_rad = 0
    doa_spherical_rad[:, 1] += azimuth_offset_rad
    doa_spherical_rad[:, 2] += elevation_offset_rad

    # Map (-pi to pi) to (0 to 1), preserving values outside range (these get wrapped in the next step)
    angles_0to1 = (doa_spherical_rad[:, 1] + np.pi) / (2 * np.pi)

    # Quantise to num_plot_angles, mapping to (0 to (num_plot_angles - 1)) with wrapping
    angles_0toN_quantised = np.round(angles_0to1 * num_plot_angles)
    angles_0toN_wrapped = angles_0toN_quantised % num_plot_angles

    angles_rad = np.linspace(-np.pi, np.pi - (2 * np.pi / num_plot_angles), num_plot_angles)
    radii = np.zeros(num_plot_angles)

    # Get energy from the omnidirectional rir channel (this is used for the radius)
    pressure = spatial_ir[start_index:end_index, 0]
    energy_linear = np.square(pressure)

    for angle_index in range(num_plot_angles):
        indices = angles_0toN_wrapped == angle_index
        radii[angle_index] = np.nansum(energy_linear[indices] * np.abs(np.cos(doa_spherical_rad[indices, 2])))

    # window_length = 5
    # radii_wrapped_for_start = radii[-window_length - 1:-1]
    # radii_wrapped_for_end = radii[:window_length]
    # radii_to_smooth = np.concat([radii_wrapped_for_start, radii, radii_wrapped_for_end])
    # radii_smoothed = savgol_filter(radii_to_smooth, window_length, 1)
    # radii_smoothed = radii_smoothed[window_length:-window_length]

    # Convert energy radius to decibels, clipping at -80 dB
    radii_dB = 10 * np.log10(np.clip(radii, 1e-8, None))

    # Mirror along the x-axis to match Treble presentation
    angles_rad_corrected = np.pi - angles_rad

    return angles_rad_corrected, radii_dB


def plotSpatioTemporalMap(spatial_rir, sample_rate, plane="median", num_plot_angles=200):
    fig, axes = plt.subplots(3, 2, subplot_kw={'projection': 'polar'})

    starts_relative_to_direct_ms = [-1, 10, 100, 200, 400, 800]

    for index, duration_ms in enumerate([3,20,20,20,20,20]):
        angles_rad, radii_dB = getSpatioTemporalMap(spatial_rir,
                                                    sample_rate,
                                                    start_ms=starts_relative_to_direct_ms[index],
                                                    duration_ms=duration_ms,
                                                    start_is_relative_to_direct=True,
                                                    plane=plane,
                                                    num_plot_angles=num_plot_angles)

        axes[index % 3][int(index / 3)].fill(angles_rad, radii_dB, color="black", alpha=1 / (index + 1))
                 # label=f"{starts_relative_to_direct_ms[index]}-{starts_relative_to_direct_ms[index] + duration_ms}")
        axes[index % 3][int(index / 3)].set_axisbelow(True)
        # axes[index % 3][int(index / 3)].legend(title='Time Region (ms)', bbox_to_anchor=(1, 1))
        axes[index % 3][int(index / 3)].set_title(f"{starts_relative_to_direct_ms[index]}-{starts_relative_to_direct_ms[index] + duration_ms}ms", x=1.6, y=0.8)
    plt.show()


def getSpatialAsymmetryScore(spatial_rir, sample_rate, show_plots=False):
    num_octave_bands = 7
    spatial_rir_octave_bands = np.zeros([num_octave_bands, len(spatial_rir[:, 0]), 4])

    for channel_index in range(4):
        octave_band_signals, octave_band_centres = Utils.getOctaveBandsFromIR(spatial_rir[:, channel_index], sample_rate)
        spatial_rir_octave_bands[:, :, channel_index] = octave_band_signals.transpose()

    num_plot_angles = 10
    num_times = 4

    all_doas = np.zeros([num_octave_bands, 3, num_times, num_plot_angles])
    circular_stds = np.zeros([num_octave_bands, 3, num_times])
    start_energies = [-25, -30, -35, -40] # dB

    for octave_band_index in range(num_octave_bands):
        spatial_rir_octave = spatial_rir_octave_bands[octave_band_index, :, :]

        # Get EDC of omni component
        edc_dB, edc_times = Energy.getEDC(spatial_rir_octave[:, 0], sample_rate)
        start_times_ms = [edc_times[Utils.findIndexOfClosest(edc_dB, start_energy)] * 1000 for start_energy in start_energies]

        for time_index, start_ms in enumerate(start_times_ms):
            planes = ["median", "transverse", "lateral"]

            for plane_index, plane in enumerate(planes):
                doa_angles, doa_radii = getSpatioTemporalMap(spatial_rir_octave,
                                                             sample_rate,
                                                             start_ms=start_ms,
                                                             duration_ms=300,
                                                             start_is_relative_to_direct=False,
                                                             plane=plane,
                                                             num_plot_angles=num_plot_angles)

                all_doas[octave_band_index, plane_index, time_index, :] = doa_radii - np.max(doa_radii)

                circular_stds[octave_band_index, plane_index, time_index] = Utils.circularStd(10 ** (doa_radii / 10), doa_angles)

    if show_plots:
        fig = plt.figure()
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "CMU Serif",
            "font.size": 15
        })
        plt.imshow(all_doas[3, 0, :, :].transpose(), aspect='auto')
        plt.ylabel("Angle")
        plt.yticks([0, (num_plot_angles - 1)/4, (num_plot_angles - 1)/2, 3 * (num_plot_angles - 1) / 4, (num_plot_angles - 1)], ["0","$\pi/2$","$\pi$","$3\pi/2$","$2\pi$"])
        plt.xticks([0,num_times/5,2*num_times/5,3*num_times/5,4*num_times/5,num_times], ["0","-10","-20","-30","-40","-50"])
        plt.xlabel("Energy Bin Start (dB)")
        plt.colorbar(location="top")
        plt.clim(-22, 0)
        fig.set_size_inches(5, 5.2)
        plt.show()

    asymmetry_score = -np.sum(circular_stds[:, 0, :])

    asymmetry_score = (asymmetry_score + 65) / 30

    return asymmetry_score
