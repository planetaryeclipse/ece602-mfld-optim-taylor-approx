import re

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch

_trial_filename_parser = re.compile(
    r"metric_(?P<metric>.+)__scaling_(?P<scaling>[\d.]+)__trial_(?P<trial>\d+)__geod_method_(?P<geod_method>\w+).pt")

@dataclass
class ParamData:
    alg: str
    metric: str
    scaling: float
    geod_method: str
    trials: List[object]


def find_param_data(data: List[ParamData], alg: str, metric: str, scaling: float, geod_method: str) -> Optional[
    ParamData]:
    for param_data in data:
        if (param_data.alg == alg
                and param_data.metric == metric
                and param_data.scaling == scaling
                and param_data.geod_method == geod_method):
            return param_data
    return None

def load_all_param_data(alg_dirs: List[Path]) -> List[ParamData]:
    data = []

    for alg_dir in alg_dirs:
        alg = alg_dir.name  # name of employed algorithm

        # gets all the files
        data_files = []
        for child in alg_dir.iterdir():
            if child.is_dir() or (child.is_file() and child.name.endswith(".gitkeep")):
                continue
            data_files.append(child)
        data_files.sort()  # ensure the initial points for each trial remain the same

        for file in data_files:
            trial_data = torch.load(file, weights_only=False)  # fails to load if weights_only=True
            trial_params_match = _trial_filename_parser.match(file.name)
            # noinspection PyUnresolvedReferences
            metric, scaling, geod_method = (trial_params_match["metric"],
                                            float(trial_params_match["scaling"]),
                                            trial_params_match["geod_method"])

            param_data = find_param_data(data, alg, metric, scaling, geod_method)
            if param_data is not None:
                param_data.trials.append(trial_data)
            else:
                param_data = ParamData(alg=alg,
                                       metric=metric,
                                       scaling=scaling,
                                       geod_method=geod_method,
                                       trials=[trial_data])
                data.append(param_data)

    return data