import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, ttest_1samp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_FILEPATH = "/Users/willcassidy/Development/GitHub/AAESUnpleasantnessModel/Data/all_results.txt"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SVRModels")

# Order matches the colouration_score, asymmetry_score, flutter_echo_score,
# curvature_score, hf_damping_score arguments of predictUnpleasantnessFromFeaturesSVR
FEATURE_NAMES = ["colouration", "flutter_echo", "curvature", "hf_damping"]  # "asymmetry" excluded

# Uncomment this for coarse search
# PARAM_GRID = {
#     "svr__kernel": ['rbf', 'linear', 'poly'],
#     "svr__C": [0.1, 1, 10, 100, 1000],
#     "svr__gamma": ['scale', 'auto', 0.001, 0.01, 0.1, 1],
#     "svr__epsilon": [0.001, 0.01, 0.1, 0.2, 0.5, 1]
# }

# Uncomment this for fine search
PARAM_GRID = {
    "svr__kernel": ['rbf'],
    "svr__C": [1, 2, 4, 6],
    "svr__gamma": [0.001, 0.005, 0.01, 0.05, 0.1],
    "svr__epsilon": [2, 4, 6, 8, 10]
}


def loadProgItemData(prog_item):
    df = pd.read_csv(DATA_FILEPATH)
    df = df[df["prog_item"] == prog_item]

    X = df[FEATURE_NAMES].values
    y = df["rating"].values
    groups = df["stimulus_id"].values
    rooms = df["room_size"].values

    return X, y, groups, rooms


def searchSVR(X_train, y_train, X_test, y_test):
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    test_fold = np.concatenate([-np.ones(len(X_train), dtype=int),
                                 np.zeros(len(X_test), dtype=int)])

    pipeline = make_pipeline(StandardScaler(), SVR())
    search = GridSearchCV(pipeline, PARAM_GRID, cv=PredefinedSplit(test_fold), scoring="r2", n_jobs=-1)
    search.fit(X_combined, y_combined)

    return search


def fitSVR(X, y, params):
    pipeline = make_pipeline(StandardScaler(), SVR())
    pipeline.set_params(**params)
    pipeline.fit(X, y)

    return pipeline


# Plots the mean cross-validated R^2 for every combination in PARAM_GRID, as one figure
# per kernel. Each figure shows a grid of heatmaps (one per epsilon value) of C vs gamma,
# with the chosen combination highlighted on the matching kernel's figure
def plotGridSearchResults(search, prog_item):
    results = pd.DataFrame(search.cv_results_)
    best_params = search.best_params_

    kernel_values = PARAM_GRID["svr__kernel"]
    c_values = PARAM_GRID["svr__C"]
    gamma_values = PARAM_GRID["svr__gamma"]
    epsilon_values = PARAM_GRID["svr__epsilon"]

    for kernel in kernel_values:
        fig, axes = plt.subplots(1, len(epsilon_values), figsize=(4 * len(epsilon_values), 4), sharey=True)

        for ax, epsilon in zip(axes, epsilon_values):
            score_grid = np.zeros((len(c_values), len(gamma_values)))

            for i, c in enumerate(c_values):
                for j, gamma in enumerate(gamma_values):
                    row = results[(results["param_svr__kernel"] == kernel)
                                  & (results["param_svr__C"] == c)
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

            if best_params["svr__kernel"] == kernel and best_params["svr__epsilon"] == epsilon:
                best_row = c_values.index(best_params["svr__C"])
                best_col = gamma_values.index(best_params["svr__gamma"])
                ax.add_patch(plt.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, fill=False,
                                           edgecolor="red", linewidth=3))

        axes[0].set_ylabel("C")
        fig.suptitle(f"Prog item {prog_item}: SVR grid search ($R^2$), kernel={kernel}, "
                     f"best = {best_params}")
        plt.tight_layout()
        plt.show()


def plotMaxPredictions(sets):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(5, 4), dpi=150)
    fig.set_layout_engine("tight")

    x_range = np.array([0, 100])
    plt.plot(x_range, x_range, linestyle='--', color='black', linewidth=0.8, dashes=(10, 5))

    for s in sets:
        ids_per_pi = {1: s["ids_pi1"], 2: s["ids_pi2"]}
        n = len(s["ids_pi1"])
        max_true = np.array([np.max([s["all_true"][p][s["all_groups"][p] == ids_per_pi[p][i]].mean()
                                      for p in [1, 2]])
                              for i in range(n)])
        max_pred = np.array([np.max([s["all_pred"][p][s["all_groups"][p] == ids_per_pi[p][i]].mean()
                                      for p in [1, 2]])
                              for i in range(n)])

        gradient, y_intercept, r_value, _, _ = linregress(max_true, max_pred)
        spearman_r, _ = spearmanr(max_true, max_pred)
        regression_line = np.poly1d([gradient, y_intercept])

        label = rf"{s['label']}: $R^2$={r_value**2:.2g}, $r_s$={spearman_r:.2g}"

        plt.plot(x_range, regression_line(x_range), color=s["color"], linewidth=1.0, alpha=0.7)
        plt.plot(max_true, max_pred, marker=s["marker"], markersize=s["markersize"], linewidth=0,
                 color=s["color"], label=label)

    plt.xlabel("Mean Unpleasantness Rating", fontsize=16)
    plt.ylabel("Predicted Unpleasantness Rating", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.legend(fontsize=12, loc="upper left", handletextpad=0.3, frameon=False)
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.show()


def printModelInfo(model, X):
    svr = model.named_steps["svr"]
    print(f"  Support vectors: {svr.n_support_[0]} / {len(X)} training samples")


def evaluateModel(model, X, y):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    spearman_correlation, _ = spearmanr(y, predictions)
    return rmse, r2, spearman_correlation


# Hyperparameters are selected using the validation set. K-fold models are trained on the
# training set only, grouped by room so each fold holds out one room. The full model is also
# trained on the training set only and returned for saving after the plot in generateAllModels.
# Returns (y_true, y_pred, groups, full_model) for the test stimuli and the fitted model
def generateModelsForProgItem(prog_item, train_stimulus_ids, val_stimulus_ids, test_stimulus_ids):
    X, y, groups, rooms = loadProgItemData(prog_item)

    train_mask = np.isin(groups, train_stimulus_ids)
    val_mask = np.isin(groups, val_stimulus_ids)
    test_mask = np.isin(groups, test_stimulus_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    full_search = searchSVR(X_train, y_train, X_val, y_val)
    best_params = full_search.best_params_
    print(f"Prog item {prog_item}: best params = {best_params}")

    plotGridSearchResults(full_search, prog_item)

    # Leave-one-room-out: fold index i always holds out room i, regardless of room sizes/order
    rooms_train = rooms[train_mask]
    held_out_rooms = np.unique(rooms_train)

    for fold_index, held_out_room in enumerate(held_out_rooms):
        val_indices = np.flatnonzero(rooms_train == held_out_room)
        train_indices = np.flatnonzero(rooms_train != held_out_room)

        model = fitSVR(X_train[train_indices], y_train[train_indices], best_params)

        rmse, r2, spearman_correlation = evaluateModel(model, X_train[val_indices], y_train[val_indices])
        print(f"Prog item {prog_item}, fold {fold_index + 1} (room {held_out_room}):")
        print(f"  Held-out RMSE = {rmse:.3f}, R^2 = {r2:.3f}, Spearman = {spearman_correlation:.3f}")
        printModelInfo(model, X_train[train_indices])

        joblib.dump(model, os.path.join(MODEL_DIR, f"prog_item_{prog_item}_fold_{fold_index + 1}.joblib"))

    full_model = fitSVR(X_train, y_train, best_params)
    y_pred_test = full_model.predict(X_test)
    rmse, r2, spearman_correlation = evaluateModel(full_model, X_test, y_test)
    print(f"Prog item {prog_item}, full model "
          f"({len(train_stimulus_ids)} train / {len(val_stimulus_ids)} val / {len(test_stimulus_ids)} test stimuli):")
    print(f"  Test RMSE = {rmse:.3f}, R^2 = {r2:.3f}, Spearman = {spearman_correlation:.3f}")
    printModelInfo(full_model, X_train)

    return (y_train, full_model.predict(X_train), groups[train_mask],
            y_val, full_model.predict(X_val), groups[val_mask],
            y_test, y_pred_test, groups[test_mask],
            full_model, best_params)


def holmBonferroni(p_values):
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    n = len(p_values)

    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, p_values[idx] * (n - rank))
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def significanceStars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plotModelInterpretation(model, X, y, prog_item):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    # Permutation importance: drop in R^2 when each feature is randomly shuffled
    perm = permutation_importance(model, X, y, n_repeats=30, scoring="r2", random_state=42, n_jobs=-1)
    sorted_indices = np.argsort(perm.importances_mean)

    # One-sided test per feature (importance > 0), Holm-Bonferroni corrected across features
    p_values = [ttest_1samp(perm.importances[i], 0, alternative="greater").pvalue
                for i in range(len(FEATURE_NAMES))]
    adjusted_p_values = holmBonferroni(p_values)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    fig.set_layout_engine("tight")
    ax.barh(range(len(FEATURE_NAMES)), perm.importances_mean[sorted_indices],
            xerr=perm.importances_std[sorted_indices], color="steelblue", ecolor="black", capsize=3)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_indices], fontsize=13)
    ax.set_xlabel("Permutation importance ($R^2$ decrease)", fontsize=13)
    ax.set_title(f"Prog item {prog_item}", fontsize=13)

    for i, idx in enumerate(sorted_indices):
        value = perm.importances_mean[idx]
        stars = significanceStars(adjusted_p_values[idx])
        ax.text(value + perm.importances_std[idx] + ax.get_xlim()[1] * 0.01,
                i, f"{float(f'{value:.3g}')} {stars}", va="center", fontsize=11)

    ax.margins(x=0.2)
    plt.show()

    # Partial dependence: marginal effect of each feature on prediction (others held at mean)
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(3 * len(FEATURE_NAMES), 3.5), dpi=150)
    fig.set_layout_engine("tight")
    PartialDependenceDisplay.from_estimator(model, X, features=range(len(FEATURE_NAMES)),
                                            feature_names=FEATURE_NAMES, ax=axes, line_kw={"color": "steelblue"})
    for ax in axes:
        ax.set_title(ax.get_title(), fontsize=11)
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
    fig.suptitle(f"Partial dependence — prog item {prog_item}", fontsize=13)
    plt.show()


def generateAllModels(test_size=0.3, val_size=0.1, random_state=42):
    df = pd.read_csv(DATA_FILEPATH)

    # Each RIR maps to one stimulus_id per programme item; split at the RIR level so both
    # programme item models use the same physical rooms in each partition
    rir_cols = ["room_size", "absorption", "rt_ratio", "loop_gain_dB", "filter", "routing"]
    rir_to_stimulus_ids = df.groupby(rir_cols)["stimulus_id"].unique()

    n_rirs = len(rir_to_stimulus_ids)
    rng = np.random.default_rng(random_state)
    shuffled_rirs = rng.permutation(n_rirs)

    n_test = int(n_rirs * test_size)
    n_val = int(n_rirs * val_size)
    test_rir_indices = shuffled_rirs[:n_test]
    val_rir_indices = shuffled_rirs[n_test:n_test + n_val]
    train_rir_indices = shuffled_rirs[n_test + n_val:]

    pi_ids = {p: set(df[df["prog_item"] == p]["stimulus_id"].values) for p in [1, 2]}

    def stimulusIdsForProgItem(rir_indices, prog_item):
        ids = []
        for i in rir_indices:
            ids.extend(sid for sid in rir_to_stimulus_ids.iloc[i] if sid in pi_ids[prog_item])
        return np.array(ids)

    all_true_train, all_pred_train, all_groups_train = {}, {}, {}
    all_true_val, all_pred_val, all_groups_val = {}, {}, {}
    all_true_test, all_pred_test, all_groups_test = {}, {}, {}
    all_models, all_train_ids, all_best_params = {}, {}, {}

    for prog_item in [1, 2]:
        train_stimulus_ids = stimulusIdsForProgItem(train_rir_indices, prog_item)
        val_stimulus_ids = stimulusIdsForProgItem(val_rir_indices, prog_item)
        test_stimulus_ids = stimulusIdsForProgItem(test_rir_indices, prog_item)
        (y_true_train, y_pred_train, groups_train,
         y_true_val, y_pred_val, groups_val,
         y_true_test, y_pred_test, groups_test,
         model, best_params) = generateModelsForProgItem(prog_item, train_stimulus_ids, val_stimulus_ids, test_stimulus_ids)
        all_true_train[prog_item] = y_true_train
        all_pred_train[prog_item] = y_pred_train
        all_groups_train[prog_item] = groups_train
        all_true_val[prog_item] = y_true_val
        all_pred_val[prog_item] = y_pred_val
        all_groups_val[prog_item] = groups_val
        all_true_test[prog_item] = y_true_test
        all_pred_test[prog_item] = y_pred_test
        all_groups_test[prog_item] = groups_test
        all_models[prog_item] = model
        all_train_ids[prog_item] = train_stimulus_ids
        all_best_params[prog_item] = best_params

    plotMaxPredictions([
        {"label": "Train", "marker": "o", "color": "black", "markersize": 3.5,
         "all_true": all_true_train, "all_pred": all_pred_train, "all_groups": all_groups_train,
         "ids_pi1": stimulusIdsForProgItem(train_rir_indices, 1),
         "ids_pi2": stimulusIdsForProgItem(train_rir_indices, 2)},
        {"label": "Validation", "marker": "s", "color": "darkblue", "markersize": 3.5,
         "all_true": all_true_val, "all_pred": all_pred_val, "all_groups": all_groups_val,
         "ids_pi1": stimulusIdsForProgItem(val_rir_indices, 1),
         "ids_pi2": stimulusIdsForProgItem(val_rir_indices, 2)},
        {"label": "Test", "marker": "+", "color": "orangered", "markersize": 5.5,
         "all_true": all_true_test, "all_pred": all_pred_test, "all_groups": all_groups_test,
         "ids_pi1": stimulusIdsForProgItem(test_rir_indices, 1),
         "ids_pi2": stimulusIdsForProgItem(test_rir_indices, 2)},
    ])

    for prog_item in [1, 2]:
        X, y, groups, _ = loadProgItemData(prog_item)
        X_train = X[np.isin(groups, all_train_ids[prog_item])]
        y_train = y[np.isin(groups, all_train_ids[prog_item])]
        model = all_models[prog_item]
        print(f"Prog item {prog_item}, final model (training data only):")
        printModelInfo(model, X_train)
        plotModelInterpretation(model, X_train, y_train, prog_item)

        # Saved model is retrained on all data (train + val + test) so that
        # predictUnpleasantnessFromFeaturesSVR(..., k_fold=-1) uses every available sample
        final_model = fitSVR(X, y, all_best_params[prog_item])
        printModelInfo(final_model, X)
        joblib.dump(final_model, os.path.join(MODEL_DIR, f"prog_item_{prog_item}_full.joblib"))


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    generateAllModels()