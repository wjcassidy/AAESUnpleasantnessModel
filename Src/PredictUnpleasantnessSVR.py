import os
import sys

import joblib
import numpy as np
from scipy.io import wavfile

sys.path.insert(1, "/Users/willcassidy/Development/GitHub/AAESOptimiser/Src/")
import Colouration
import DSE
import FlutterEcho
import HFDamping
import SDM

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVRModels")

_model_cache = {}


def _loadModel(prog_item, k_fold=-1):
    if k_fold == -1:
        filename = f"prog_item_{prog_item}_full.joblib"
    else:
        filename = f"prog_item_{prog_item}_fold_{k_fold}.joblib"

    filepath = os.path.join(MODEL_DIR, filename)

    if filepath not in _model_cache:
        _model_cache[filepath] = joblib.load(filepath)

    return _model_cache[filepath]


def predictUnpleasantnessFromRIR(rir_filepath):
    sample_rate, spatial_rir = wavfile.read(rir_filepath)
    spatial_rir = spatial_rir / np.max(np.abs(spatial_rir))

    colouration_score = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
    asymmetry_score = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
    flutter_echo_score = FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, False)
    curvature_score = DSE.getCurvature(spatial_rir[:, 0], sample_rate)
    hf_damping_score = HFDamping.getHFDampingScore(spatial_rir[:, 0], sample_rate)

    return predictUnpleasantnessFromFeaturesSVR(colouration_score, asymmetry_score, flutter_echo_score,
                                                 curvature_score, hf_damping_score, 0, use_max_from_both=True)


# Order of arguments matches FEATURE_NAMES in GenerateSVR.py
def predictUnpleasantnessFromFeaturesSVR(colouration_score, asymmetry_score, flutter_echo_score, curvature_score,
                                         hf_damping_score, prog_item, k_fold=-1, use_max_from_both=False):
    if use_max_from_both:
        prog_item_1_unpleas = predictUnpleasantnessFromFeaturesSVR(colouration_score, asymmetry_score,
                                                                    flutter_echo_score, curvature_score,
                                                                    hf_damping_score, 1, k_fold=k_fold,
                                                                    use_max_from_both=False)
        prog_item_2_unpleas = predictUnpleasantnessFromFeaturesSVR(colouration_score, asymmetry_score,
                                                                    flutter_echo_score, curvature_score,
                                                                    hf_damping_score, 2, k_fold=k_fold,
                                                                    use_max_from_both=False)
        return np.max([prog_item_1_unpleas, prog_item_2_unpleas])

    if prog_item not in (1, 2):
        assert False

    model = _loadModel(prog_item, k_fold)

    features = np.array([[colouration_score, asymmetry_score, flutter_echo_score, curvature_score,
                           hf_damping_score]])

    return model.predict(features)[0]


if __name__ == "__main__":
    ir_dir = "/Users/willcassidy/Development/GitHub/PyRES-ClosedLoop/data/AudioExamples/LargeSystemRoom1EnvFOA/init_foa_test_R0_S0.wav"
    sample_rate, spatial_rir = wavfile.read(ir_dir)
    spatial_rir = spatial_rir / np.max(np.abs(spatial_rir))

    print(f"Spatial Asym: {SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)}")
    print(f"Flutter Echo: {FlutterEcho.getFlutterEchoScore(spatial_rir, sample_rate, False)}")
    print(f"Colouration: {Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)}")
    print(f"HF Damping: {HFDamping.getHFDampingScore(spatial_rir[:, 0], sample_rate, False)}")
    print(f"DSE Curvature: {DSE.getCurvature(spatial_rir[:, 0], sample_rate)}")
    print(f"\nOverall Unpleasantness (SVR): {predictUnpleasantnessFromRIR(ir_dir)}")