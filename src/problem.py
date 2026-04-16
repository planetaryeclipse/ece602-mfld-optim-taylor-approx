import torch

from typing import Optional, Tuple, List

from diff_mfld.geometry.funcs import MfldFunc, RiemannSquaredDist, FuncArgs
from diff_mfld.mfld import ComputeMfld


class TargetCost(MfldFunc):
    def __init__(self, target: torch.Tensor):
        self._target = target
        self._dist_func = 0.5 * RiemannSquaredDist()

    @property
    def target(self):
        return self._target

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._dist_func.value(p, cfg, self._target, *args)

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._dist_func.diff(p, cfg, self._target, *args)

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._dist_func.hess(p, cfg, self._target, *args)


def create_problem(cost_target: torch.Tensor,
                   ineq_constr_region: Optional[List[List[Tuple[torch.Tensor, float]]]] = None
                   ) -> Tuple[TargetCost, Optional[MfldFunc]]:
    cost_func = TargetCost(cost_target)

    return cost_func, None  # for now
