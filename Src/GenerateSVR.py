import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_FILEPATH = "/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel/Data/all_results.txt"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVRModels")

# Order matches the colouration_score, asymmetry_score, flutter_echo_score,
# curvature_score, hf_damping_score arguments of predictUnpleasantnessFromFeaturesSVR
FEATURE_NAMES = ["colouration", "asymmetry", "flutter_echo", "curvature", "hf_damping"]

PARAM_GRID = {
    "svr__kernel": ["rbf"],
    "svr__C": [2, 4, 6],
    "svr__gamma": ["scale", "auto", 0.1, 1],
    "svr__epsilon": [0.5, 1, 2, 5],
}


def loadProgItemData(prog_item):
    df = pd.read_csv(DATA_FILEPATH)
    df = df[df["prog_item"] == prog_item]

    X = df[FEATURE_NAMES].values
    y = df["rating"].values
    groups = df["stimulus_id"].values

    return X, y, groups


def searchSVR(X, y, groups, n_splits=3):
    pipeline = make_pipeline(StandardScaler(), SVR())

    search = GridSearchCV(pipeline, PARAM_GRID, cv=GroupKFold(n_splits=n_splits), scoring="r2", n_jobs=-1)
    search.fit(X, y, groups=groups)

    return search


def fitSVR(X, y, params):
    pipeline = make_pipeline(StandardScaler(), SVR())
    pipeline.set_params(**params)
    pipeline.fit(X, y)

    return pipeline


# Plots the mean cross-validated R^2 for every combination in PARAM_GRID, as a grid of
# heatmaps (one per epsilon value), with the chosen combination highlighted
def plotGridSearchResults(search, prog_item):
    results = pd.DataFrame(search.cv_results_)
    best_params = search.best_params_

    c_values = PARAM_GRID["svr__C"]
    gamma_values = PARAM_GRID["svr__gamma"]
    epsilon_values = PARAM_GRID["svr__epsilon"]

    fig, axes = plt.subplots(1, len(epsilon_values), figsize=(4 * len(epsilon_values), 4), sharey=True)

    for ax, epsilon in zip(axes, epsilon_values):
        score_grid = np.zeros((len(c_values), len(gamma_values)))

        for i, c in enumerate(c_values):
            for j, gamma in enumerate(gamma_values):
                row = results[(results["param_svr__C"] == c)
                              & (results["param_svr__gamma"] == gamma)
                              & (results["param_svr__epsilon"] == epsilon)]
                score_grid[i, j] = row["mean_test_score"].values[0]

        im = ax.imshow(score_grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(gamma_values)))
        ax.set_xticklabels(gamma_values)
        ax.set_yticks(range(len(c_values)))
        ax.set_yticklabels(c_values)
        ax.set_xlabel("gamma")
        ax.set_title(f"epsilon = {epsilon}")
        fig.colorbar(im, ax=ax)

        if best_params["svr__epsilon"] == epsilon:
            best_row = c_values.index(best_params["svr__C"])
            best_col = gamma_values.index(best_params["svr__gamma"])
            ax.add_patch(plt.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, fill=False,
                                       edgecolor="red", linewidth=3))

    axes[0].set_ylabel("C")
    fig.suptitle(f"Prog item {prog_item}: SVR grid search (R^2), full dataset, "
                 f"best = {best_params}")
    plt.tight_layout()
    plt.show()


def evaluateModel(model, X, y):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    spearman_correlation, _ = spearmanr(y, predictions)
    return rmse, r2, spearman_correlation


# Grid search is run only on the full dataset, giving a single hyperparameter choice for the
# programme item. The k-fold splits then only re-fit with those fixed hyperparameters, purely
# to evaluate held-out performance, mirroring the k-fold scheme used by the MLR model
def generateModelsForProgItem(prog_item):
    X, y, groups = loadProgItemData(prog_item)

    full_search = searchSVR(X, y, groups)
    best_params = full_search.best_params_
    print(f"Prog item {prog_item}: best params = {best_params}")

    plotGridSearchResults(full_search, prog_item)

    group_kfold = GroupKFold(n_splits=3)

    for fold_index, (train_indices, test_indices) in enumerate(group_kfold.split(X, y, groups)):
        model = fitSVR(X[train_indices], y[train_indices], best_params)

        rmse, r2, spearman_correlation = evaluateModel(model, X[test_indices], y[test_indices])
        print(f"Prog item {prog_item}, fold {fold_index + 1}:")
        print(f"  Held-out RMSE = {rmse:.3f}, R^2 = {r2:.3f}, Spearman = {spearman_correlation:.3f}")

        joblib.dump(model, os.path.join(MODEL_DIR, f"prog_item_{prog_item}_fold_{fold_index + 1}.joblib"))

    full_model = fitSVR(X, y, best_params)
    rmse, r2, spearman_correlation = evaluateModel(full_model, X, y)
    print(f"Prog item {prog_item}, full model:")
    print(f"  Training RMSE = {rmse:.3f}, R^2 = {r2:.3f}, Spearman = {spearman_correlation:.3f}")

    joblib.dump(full_model, os.path.join(MODEL_DIR, f"prog_item_{prog_item}_full.joblib"))


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    for prog_item in [1, 2]:
        generateModelsForProgItem(prog_item)