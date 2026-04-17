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
        return self._dist_func.value(p, cfg, self._target)

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._dist_func.diff(p, cfg, self._target)

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._dist_func.hess(p, cfg, self._target)


class ConstrainedRegion(MfldFunc):
    def __init__(self, regions: List[Tuple[torch.Tensor, float]]):
        self._regions = regions
        self._dist_fn = RiemannSquaredDist()

    @property
    def regions(self):
        return self._regions

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        region_values = torch.zeros((len(self._regions)))
        for i, (center, radius) in enumerate(self._regions):
            region_values[i] = self._dist_fn.value(p, cfg, center).item() - radius

        return torch.max(region_values)

    # noinspection DuplicatedCode
    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        region_values = torch.zeros((len(self._regions)))
        for i, (center, radius) in enumerate(self._regions):
            region_values[i] = self._dist_fn.value(p, cfg, center).item() - radius

        max_idx = torch.argmax(region_values)
        center, _ = self._regions[max_idx]
        return self._dist_fn.diff(p, cfg, center)

    # noinspection DuplicatedCode
    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        region_values = torch.zeros((len(self._regions)))
        for i, (center, radius) in enumerate(self._regions):
            region_values[i] = self._dist_fn.value(p, cfg, center).item() - radius

        max_idx = torch.argmax(region_values)
        center, _ = self._regions[max_idx]
        return self._dist_fn.hess(p, cfg, center)

def create_problem(cost_target: torch.Tensor,
                   ineq_constr_regions: Optional[List[List[Tuple[torch.Tensor, float]]]] = None
                   ) -> Tuple[TargetCost, List[ConstrainedRegion]]:
    cost_func = TargetCost(cost_target)  # generates the cost function

    region_ineq_funcs = []
    if ineq_constr_regions is not None:
        for region in ineq_constr_regions:
            region_ineq_funcs.append(ConstrainedRegion(region))

    return cost_func, region_ineq_funcs
