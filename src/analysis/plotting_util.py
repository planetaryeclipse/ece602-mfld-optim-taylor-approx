import numpy as np
import matplotlib.pyplot as plt

from typing import List

from analysis.util import ParamData

_ivpbvp_data_label = "ivpbvp"

_desired_method_order = [
    "ivpbvp",
    "o1",
    "o2",
    "o3",
    "o2_o1",
    "o3_o1",
    "o3_o2"
]

_method_styles = {
    "ivpbvp": ("IVP-BVP", "-"),
    "o1": ("Exp-Log O1", "--"),
    "o2": ("Exp-Log O2", "-."),
    "o3": ("Exp-Log O3", ":"),
    "o2_o1": ("Exp O2, Log O1", (0, (5, 1))),
    "o3_o1": ("Exp O3, Log O1", (0, (3, 1, 1, 1))),
    "o3_o2": ("Exp O3, Log O2", (0, (1, 1))),
}

_max_success_lim = 21

_cost_target = np.array([-0.3477236, -0.06954472, 0.20863416])
_constraint_target = np.array([-0.13908944,  0.20863416,  0.06954472])


def gen_success_vs_scaling_plots(data: List[ParamData], alg: str, metric: str, scaling_to_show: List[float],
                                 methods_to_show: List[str], show_titles=True, method_styles=_method_styles):
    alg_and_metric_data = list(filter(lambda param_data: param_data.alg == alg and param_data.metric == metric, data))

    fig, ax = plt.subplots()

    for method in methods_to_show:
        successes_vs_scaling = []
        method_data = filter(lambda param_data: param_data.geod_method == method, alg_and_metric_data)
        for scaling in scaling_to_show:
            trials = next(param_data for param_data in method_data if param_data.scaling == scaling).trials
            num_successes = 0
            for trial in trials:
                if trial.success:
                    num_successes += 1
            successes_vs_scaling.append(num_successes)

        ax.plot(scaling_to_show, successes_vs_scaling,
                label=_method_styles[method][0],
                alpha=0.7,
                linewidth=2,
                linestyle=_method_styles[method][1])

    if show_titles:
        ax.title.set_text(f"alg: {alg}, metric: {metric}")

    ax.set_xlabel("Scaling")
    ax.set_ylabel("Successes")
    ax.set_ylim(0, _max_success_lim)
    ax.legend()


def gen_iters_vs_scaling_plots(data: List[ParamData], alg: str, metric: str, scaling_to_show: List[float],
                               methods_to_show: List[str], show_titles=True, method_styles=_method_styles):
    alg_and_metric_data = list(filter(lambda param_data: param_data.alg == alg and param_data.metric == metric, data))

    fig, ax = plt.subplots()

    overall_max_iters = 0

    for method in methods_to_show:
        mean_iters_vs_scaling = np.zeros(len(scaling_to_show))
        std_iters_vs_scaling = np.zeros(len(scaling_to_show))

        method_data = filter(lambda param_data: param_data.geod_method == method, alg_and_metric_data)
        for i, scaling in enumerate(scaling_to_show):
            trials = next(param_data for param_data in method_data if param_data.scaling == scaling).trials

            num_iters = []
            for trial in trials:
                if trial.success:
                    num_iters.append(trial.iters)

            if len(num_iters) == 0:
                continue

            max_iters = int(np.max(num_iters))
            if max_iters > overall_max_iters:
                overall_max_iters = max_iters

            iters_mean = np.mean(num_iters)
            iters_std = np.std(num_iters)

            mean_iters_vs_scaling[i] = iters_mean
            std_iters_vs_scaling[i] = iters_std

        line, = ax.plot(scaling_to_show, mean_iters_vs_scaling,
                        label=_method_styles[method][0],
                        alpha=0.7,
                        linewidth=2,
                        linestyle=_method_styles[method][1])
        line_color = line.get_color()

        plt.fill_between(scaling_to_show, mean_iters_vs_scaling + std_iters_vs_scaling, mean_iters_vs_scaling,
                         color=line_color, alpha=0.2)

    if show_titles:
        ax.title.set_text(f"alg: {alg}, metric: {metric}")

    ax.set_xlabel("Scaling")
    ax.set_ylabel("Iterations")
    ax.set_ylim(0, overall_max_iters + 1)
    ax.legend()


def gen_euclid_nonoptimal_norms_vs_scaling_plots(data: List[ParamData], alg: str, metric: str,
                                                 scaling_to_show: List[float],
                                                 methods_to_show: List[str], show_titles=True,
                                                 method_styles=_method_styles,
                                                 target=_cost_target,
                                                 use_log=False):
    alg_and_metric_data = list(filter(lambda param_data: param_data.alg == alg and param_data.metric == metric, data))

    fig, ax = plt.subplots()

    for method in methods_to_show:
        mean_euclid_dist_norm_vs_scaling = np.zeros(len(scaling_to_show))
        std_euclid_dist_norm_vs_scaling = np.zeros(len(scaling_to_show))

        method_data = filter(lambda param_data: param_data.geod_method == method, alg_and_metric_data)
        for i, scaling in enumerate(scaling_to_show):
            trials = next(param_data for param_data in method_data if param_data.scaling == scaling).trials

            euclid_dist_norms = []
            for trial in trials:
                if trial.success:
                    euclid_dist_norm = np.linalg.norm(scaling * target - np.array(trial.p))
                    euclid_dist_norms.append(euclid_dist_norm)

            if len(euclid_dist_norms) == 0:
                continue

            iters_mean = np.mean(euclid_dist_norms)
            iters_std = np.std(euclid_dist_norms)

            mean_euclid_dist_norm_vs_scaling[i] = iters_mean
            std_euclid_dist_norm_vs_scaling[i] = iters_std

        mean_line = mean_euclid_dist_norm_vs_scaling
        std_line = mean_euclid_dist_norm_vs_scaling + std_euclid_dist_norm_vs_scaling

        if use_log:
            mean_line = np.log(mean_line)
            std_line = np.log(std_line)

        line, = ax.plot(scaling_to_show, mean_line,
                        label=_method_styles[method][0],
                        alpha=0.7,
                        linewidth=2,
                        linestyle=_method_styles[method][1])
        line_color = line.get_color()

        plt.fill_between(scaling_to_show,
                         std_line,
                         mean_line,
                         color=line_color, alpha=0.2)

    if show_titles:
        ax.title.set_text(f"alg: {alg}, metric: {metric}")

    ax.set_xlabel("Scaling")
    ax.set_ylabel("Euclid Distance Norm")
    # ax.set_ylim(0, _max_success_lim)
    ax.legend()


def gen_euclid_nonoptimal_norms_plots(data: List[ParamData], alg: str, metric: str,
                                      scaling: float,
                                      methods_to_show: List[str], show_titles=True,
                                      method_styles=_method_styles,
                                      target=_cost_target,
                                      use_log=False):
    methods_data = list(filter(
        lambda param_data: param_data.alg == alg and param_data.metric == metric and param_data.scaling == scaling,
        data))

    fig, ax = plt.subplots()

    for method in methods_to_show:

        trials = next(param_data for param_data in methods_data if param_data.geod_method == method).trials
        successful_trials = list(filter(lambda trial: trial.success, trials))

        max_iter_len_for_method = max(trial.iters for trial in successful_trials)
        euclid_dist_norms = np.zeros((len(successful_trials), max_iter_len_for_method))

        for i, trial in enumerate(successful_trials):
            p_hist = trial.history.p_hist
            len_trial = p_hist.shape[0]

            curr_dist_norm = None
            for j in range(max_iter_len_for_method):
                # if this iteration terminated earlier then we just keep the final point
                curr_dist_norm = np.linalg.norm(
                    scaling * target - np.array(p_hist[j, :])) if j < len_trial else curr_dist_norm
                euclid_dist_norms[i, j] = curr_dist_norm

        mean_euclid_dist_norm = np.mean(euclid_dist_norms, axis=0)
        std_euclid_dist_norm = np.std(euclid_dist_norms, axis=0)

        mean_line = mean_euclid_dist_norm
        std_line = mean_euclid_dist_norm + std_euclid_dist_norm

        if use_log:
            mean_line = np.log(mean_line)
            std_line = np.log(std_line)

        iters = list(range(1, max_iter_len_for_method + 1))

        line, = ax.plot(iters, mean_line,
                        label=_method_styles[method][0],
                        alpha=0.7,
                        linewidth=2,
                        linestyle=_method_styles[method][1])
        line_color = line.get_color()

        plt.fill_between(iters,
                         std_line,
                         mean_line,
                         color=line_color, alpha=0.2)

    if show_titles:
        ax.title.set_text(f"alg: {alg}, metric: {metric}, scaling: {scaling}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Euclid Distance Norm")
    # ax.set_ylim(0, _max_success_lim)
    ax.legend()


# plotting methods exclusive to the constrained solver

def gen_subsolver_iters_plots(data: List[ParamData], alg: str, metric: str,
                              scaling: float,
                              methods_to_show: List[str], show_titles=True,
                              method_styles=_method_styles,
                              use_log=False):
    methods_data = list(filter(
        lambda param_data: param_data.alg == alg and param_data.metric == metric and param_data.scaling == scaling,
        data))

    fig, ax = plt.subplots()

    overall_max_subsolver_iters = 0

    for method in methods_to_show:

        trials = next(param_data for param_data in methods_data if param_data.geod_method == method).trials
        successful_trials = list(filter(lambda trial: trial.success, trials))

        max_iter_len_for_method = max(trial.iters for trial in successful_trials)
        subsolver_iters = np.zeros((len(successful_trials), max_iter_len_for_method))

        for i, trial in enumerate(successful_trials):
            p_hist = trial.history.p_hist
            len_trial = p_hist.shape[0]

            curr_subsolver_iters = None
            for j in range(max_iter_len_for_method):
                # if this iteration terminated earlier then we just keep the final point
                curr_subsolver_iters = trial.history.subsolver_iters_hist[j] if j < len_trial else curr_subsolver_iters
                subsolver_iters[i, j] = curr_subsolver_iters

        max_subsolver_iters = int(np.max(subsolver_iters))
        if max_subsolver_iters > overall_max_subsolver_iters:
            overall_max_subsolver_iters = max_subsolver_iters

        mean_subsolver_iters = np.mean(subsolver_iters, axis=0)
        std_subsolver_iters = np.std(subsolver_iters, axis=0)

        iters = list(range(max_iter_len_for_method))

        line, = ax.plot(iters, mean_subsolver_iters,
                        label=_method_styles[method][0],
                        alpha=0.7,
                        linewidth=2,
                        linestyle=_method_styles[method][1])
        line_color = line.get_color()

        plt.fill_between(iters,
                         mean_subsolver_iters + std_subsolver_iters,
                         mean_subsolver_iters,
                         color=line_color, alpha=0.2)

    if show_titles:
        ax.title.set_text(f"alg: {alg}, metric: {metric}, scaling: {scaling}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Subsolver Iterations")
    ax.set_ylim(0, overall_max_subsolver_iters + 1)
    ax.legend()
