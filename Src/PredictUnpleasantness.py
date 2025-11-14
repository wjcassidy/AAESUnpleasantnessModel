from scipy.io import wavfile
import Colouration
import FlutterEcho
import numpy as np
from scipy import stats
import SDM
import DSE
import HFDamping
from scipy import signal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile


# Reads the RIR files in folder "Labelled {feature}", the names of which are ranked from 0-10
# (e.g. "0.wav", "0_1.wav", "1.wav"), and compares these to the feature outputs for the RIRs.
# feature = "Colouration" | "Spatial Asymmetry" | "Flutter Echo"
def evaluateFeature(feature="Colouration", show_stimulus_ids=False):
    feature_rirs_dir = f"/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel/Audio/{feature}/"
    stimulus_filenames = [filename for filename in listdir(feature_rirs_dir) if isfile(feature_rirs_dir + filename) and filename.endswith("wav")]

    results_filepath = f"/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel/FeatureListeningTest/{feature}_results.csv"

    with open(results_filepath, 'r') as file:
        results_lines = file.readlines()

    num_stimuli = len(stimulus_filenames)
    num_subjects = int(np.floor(len(results_lines) / len(stimulus_filenames)))
    results = np.zeros([num_stimuli, num_subjects]) # audio file index, subject

    stimulus_index = 0
    subject_index = 0

    for results_line in results_lines:
        results_line_split = results_line.split(",")

        if results_line_split[0] == " ":
            continue

        results[int(results_line_split[0].strip(".wav")) - 1][subject_index] = int(results_line_split[1].strip(" \n"))

        stimulus_index += 1

        if stimulus_index == num_stimuli:
            subject_index += 1
            stimulus_index = 0

    feature_outputs = np.zeros_like(stimulus_filenames)

    for filename in stimulus_filenames:
        filepath = feature_rirs_dir + filename
        file_index = int(filename.strip(".wav")) - 1
        sample_rate, spatial_rir = wavfile.read(filepath)

        if feature == "Colouration":
            feature_outputs[file_index] = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
        elif feature == "Asymmetry":
            feature_outputs[file_index] = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
        elif feature == "Flutter":
            feature_outputs[file_index] = FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, False)
        elif feature == "HFDamping":
            feature_outputs[file_index] = HFDamping.getHFDampingScore(spatial_rir[:, 0], sample_rate, False)
        else:
            assert False

    feature_outputs = [float(output) for output in feature_outputs]

    mean_results = np.mean(results, 1)
    all_results = results.flatten()
    repeated_feature_outputs = np.repeat(feature_outputs, num_subjects)

    gradient, y_intercept, r_value, p_value, std_err = stats.linregress(mean_results, feature_outputs)
    # gradient, y_intercept, r_value, p_value, std_err = stats.linregress(all_results, repeated_feature_outputs)
    spearman_correlation, spearman_sig = stats.spearmanr(mean_results, feature_outputs)
    linear_regression = np.poly1d([gradient, y_intercept])

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "CMU Serif",
        "font.size": 15
    })
    plt.plot(mean_results, feature_outputs, 'o')
    plt.plot([0, 100], linear_regression([0, 100]))
    # plt.plot([0, 100], [0, 1], linestyle='--', color='black', linewidth=0.5, dashes=(10,5))
    # plt.plot(all_results, repeated_feature_outputs, 'o', all_results, linear_regression(all_results))
    plt.xlabel(f"True Rating")
    plt.ylabel(f"Predicted")
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.title(f"{feature} ($R^2$ = {round(r_value ** 2, 2)}, Spearman's = {round(spearman_correlation, 2)})")#, p = {round(p_value, 5)})")

    if show_stimulus_ids:
        for i in range(15):
            plt.annotate(str(i + 1), (mean_results[i], feature_outputs[i]))

    plt.show()


def predictUnpleasantnessFromRIR(rir_filepath):
    sample_rate, spatial_rir = wavfile.read(rir_filepath)

    # Compute features
    colouration_score = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
    asymmetry_score = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
    flutter_echo_score = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, False)

    return predictUnpleasantnessFromFeatures(colouration_score, asymmetry_score, flutter_echo_score)


def predictUnpleasantnessFromFeatures(colouration_score, asymmetry_score, flutter_echo_score, curvature_score, spectral_score, prog_item, k_fold=-1):
    if k_fold == -1:
        k_fold_index = 3
    else:
        k_fold_index = k_fold - 1

    # First three values are from the respective k-fold, the last is trained on all data
    if prog_item == 1:
        y_intercept =          [4.014,  -6.114, -13.399, 0.441]
        colouration_gradient = [19.661, 63.282, 29.815,  29.566]
        flutter_gradient =     [14.186, 23.199, 15.996,  12.387]
        asymmetry_gradient =   [13.808, 2.134,  29.673, 23.356]
        curvature_gradient =   [25.760, 41.189, 31.543,  27.132]
        hf_damping_gradient =  [19.898, 11.748, 26.366,  16.431]
    elif prog_item == 2:
        y_intercept =          [29.475,  23.835,  15.898,  21.370]
        colouration_gradient = [56.435,  56.939,  70.117,  64.888]
        flutter_gradient =     [-4.435,  -13.447, -8.640, -9.772]
        asymmetry_gradient =   [-20.606, -6.970,  -1.446, -7.511]
        curvature_gradient =   [11.943,  36.026,  14.487, 14.986]
        hf_damping_gradient =  [-18.884, -23.701, -16.167,  -17.549]
    else:
        assert False

    linear_model = (y_intercept[k_fold_index]
                    + colouration_gradient[k_fold_index] * colouration_score
                    + asymmetry_gradient[k_fold_index] * asymmetry_score
                    + flutter_gradient[k_fold_index] * flutter_echo_score
                    + curvature_gradient[k_fold_index] * curvature_score
                    + hf_damping_gradient[k_fold_index] * spectral_score)

    return linear_model


if __name__ == "__main__":
    # filename = "Flutter.wav" # high flutter
    # filename = "Room3.wav" # pretty high flutter, front-back though
    # filename = "PassiveRoom.wav" # fairly high flutter, late
    # filename = "horizontal_1.wav" # near zero flutter
    # filename = "NoiseIR_RT_1s_48k_1.wav" # zero flutter
    # filename = "Room33.wav"
    # filename = "SingleLSLeft.wav"
    # filename = "Wet.wav"
    # filename = "BrightLate.wav"
    # filename = "BrightLateColoured.wav"
    # filename = "Colouration/15.wav"

    # Spatial RIRs
    # filename = "Asymmetry/11.wav"

    # Passive Rooms
    filename = "Passive11.wav"
    # filename = "Room3.wav"
    # filename = "PassiveRoom.wav"
    # filename = "Pilsen.wav"
    # filename = "Pilsen2.wav"
    # filename = "Pilsen3.wav"
    # filename = "Normal.wav"
    # filename = "DSE.wav"

    # filename = "Stimulus64.wav"
    # filename = "Stimulus101.wav"

    sample_rate, spatial_rir = wavfile.read(f"/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel/Audio/{filename}")
    # SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, True)
    # FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, True)
    # Colouration.getColouration(spatial_rir[:, 0], sample_rate, True)
    # HFDamping.getHFDampingScore(spatial_rir[:, 0], sample_rate, True)

    # evaluateFeature("Colouration")
    # evaluateFeature("Asymmetry")
    # evaluateFeature("Flutter")
    evaluateFeature("HFDamping")
