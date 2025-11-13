import numpy as np
import Energy
import Utils
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy import stats

def showPlots(edc_dB,
              edc_times,
              early_gradient,
              late_gradient,
              curvature):
    plt.plot(edc_times, edc_dB)
    plt.plot(edc_times, np.multiply(edc_times, early_gradient), 'b-')
    plt.plot(edc_times, np.multiply(edc_times, late_gradient), 'r-')
    plt.title(f"Curvature = {np.round(curvature, 2)}")
    plt.ylim([-60, 0])
    plt.show()

def getCurvature(rir, sample_rate, should_high_pass=True, show_plots=False):
    if should_high_pass:
        order = 4
        cutoff = 500.0
        sos = butter(order, cutoff, 'highpass', fs=sample_rate, output='sos')
        rir = sosfilt(sos, rir)

    edc_dB, edc_times = Energy.getEDC(rir, sample_rate)

    early_start_dB = -5.0
    early_end_dB = -10.0
    late_start_dB = -35.0
    late_end_dB = -40.0

    early_start_index = Utils.findIndexOfClosest(edc_dB, early_start_dB)
    early_end_index = Utils.findIndexOfClosest(edc_dB, early_end_dB)
    late_start_index = Utils.findIndexOfClosest(edc_dB, late_start_dB)
    late_end_index = Utils.findIndexOfClosest(edc_dB, late_end_dB)

    early_gradient, _, _, _, _ = stats.linregress(edc_times[early_start_index:early_end_index], edc_dB[early_start_index:early_end_index])
    late_gradient, _, _, _, _ = stats.linregress(edc_times[late_start_index:late_end_index], edc_dB[late_start_index:late_end_index])

    curvature = 1.0 - (late_gradient / early_gradient)

    if show_plots:
        showPlots(edc_dB,
                  edc_times,
                  early_gradient,
                  late_gradient,
                  curvature)

    return curvature
