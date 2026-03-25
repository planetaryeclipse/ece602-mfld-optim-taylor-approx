import torch

from copy import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union, Optional

from optim.methods import SubsolverCfg, SubsolverMethod
from optim.results import CustomConstrSolverResult, ConstrSolverResult, ConstrSolverHistory, SubsolverCfg
from src.diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from src.diff_mfld.mfld import Mfld, ComputeMfld


# implements the Riemannian Augmented Lagrangian Method (RALM) described in "Simple Algorithms for Optimization on
# Riemannian Manifolds with Constraints"

@dataclass
class RalmCfg:
    acc_tol_min: float
    acc_tol_0: float
    acc_decay: float  # in (0, 1)

    penalty_0: float
    penalty_growth: float  # > 1

    g_mult_min: torch.Tensor
    g_mult_max: torch.Tensor
    h_mult_min: torch.Tensor
    h_mult_max: torch.Tensor

    ratio: float  # in (0, 1)
    min_step: float

    subsolver_method: SubsolverMethod
    subsolver_cfg: SubsolverCfg  # must be type corresponding to the subsolver method
    max_iters: int


@dataclass
class RalmHistory:
    p_hist: torch.Tensor

    f_hist: torch.Tensor
    gs_hist: torch.Tensor
    hs_hist: torch.Tensor

    acc_hist: torch.Tensor
    penalty_hist: torch.Tensor

    g_mults_hist: torch.Tensor
    h_mults_hist: torch.Tensor


@dataclass
class RalmResult(CustomConstrSolverResult):
    success: bool
    p: torch.Tensor
    iters: int
    history: RalmHistory

    @property
    def result(self) -> ConstrSolverResult:
        return ConstrSolverResult(
            success=self.success,
            p=self.p,
            iters=self.iters,
            history=ConstrSolverHistory(
                p_hist=self.history.p_hist,
                f_hist=self.history.f_hist,
                gs_hist=self.history.gs_hist,
                hs_hist=self.history.hs_hist,
                g_mults_hist=self.history.g_mults_hist,
                h_mults_hist=self.history.h_mults_hist,
            )
        )


class AugmentedLagrangian(MfldFunc):
    def __init__(self,
                 f: MfldFunc,
                 gs: List[MfldFunc],
                 hs: List[MfldFunc],
                 g_mults: List[float],
                 h_mults: List[float],
                 penalty: float):
        self._f = f
        self._gs = gs
        self._hs = hs
        self._g_mults = g_mults
        self._h_mults = h_mults
        self._penalty = penalty

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_sum = sum(
            torch.maximum(torch.tensor(0.0), g_mult / self._penalty + g.value(p, cfg, *args)) ** 2
            for g, g_mult in zip(self._gs, self._g_mults)
        )
        h_sum = sum(
            (h.value(p, cfg, *args) + h_mult / self._penalty) ** 2
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_value = self._f.value(p, cfg, *args) + self._penalty / 2.0 * (g_sum + h_sum)
        return aug_value

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_vals = [g.value(p, cfg, *args) for g in self._gs]
        g_diff_sum = sum(
            2.0 * (g_mult / self._penalty + g_val) * g.diff(p, cfg, *args) if g_val > 0.0 else 0.0
            for g, g_mult, g_val in zip(self._gs, self._g_mults, g_vals)
        )
        h_diff_sum = sum(
            2.0 * (h_mult / self._penalty + h.value(p, cfg, *args)) * h.diff(p, cfg, *args)
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_diff_value = self._f.diff(p, cfg, *args) + self._penalty / 2.0 * (g_diff_sum + h_diff_sum)
        return aug_diff_value

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_vals = [g.value(p, cfg, *args) for g in self._gs]
        g_diff_vals = [g.diff(p, cfg, *args) for g in self._gs]

        h_vals = [h.value(p, cfg, *args) for h in self._hs]
        h_diff_vals = [h.diff(p, cfg, *args) for h in self._hs]

        g_hess_sum = sum(
            2.0 * torch.outer(g_diff, g_diff) + 2.0 * (g_val + g_mult / self._penalty) * g.hess(p, cfg, *args)
            if g_val > 0.0 else 0.0
            for g, g_mult, g_val, g_diff in zip(self._g, self._g_mults, g_vals, g_diff_vals))
        h_hess_sum = sum(
            2.0 * torch.outer(h_diff, h_diff) + 2.0 * (h_val / self._penalty) * h.hess(p, cfg, *args)
            for h, h_mult, h_val, h_diff in zip(self._h, self._h_mults, h_vals, h_diff_vals)
        )
        aug_hess_value = self._f.hess(p, cfg, *args) + self._penalty / 2.0 * (g_hess_sum + h_hess_sum)
        return aug_hess_value

    @property
    def penalty(self):
        return self._penalty

    @property
    def g_mults(self):
        return self._g_mults

    @property
    def h_mults(self):
        return self._h_mults

    @penalty.setter
    def penalty(self, penalty: float):
        self._penalty = penalty

    @g_mults.setter
    def g_mults(self, g_mults: List[float]):
        self._g_mults = g_mults

    @h_mults.setter
    def h_mults(self, h_mults: List[float]):
        self._h_mults = h_mults


def ralm(
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        p0: torch.Tensor,
        g_mults_0: torch.Tensor,
        h_mults_0: torch.Tensor,
        mfld: ComputeMfld,
        cfg: RalmCfg,
        *args: *FuncArgs
) -> RalmResult:
    g_mults = g_mults_0
    h_mults = h_mults_0

    p = p0.clone()
    sigma: Optional[torch.Tensor] = None  # no value at start

    # track the various histories over time
    p_hist = []
    f_hist = []
    gs_hist = []
    hs_hist = []
    acc_hist = []
    penalty_hist = []
    g_mults_hist = []
    h_mults_hist = []

    penalty = cfg.penalty_0
    acc_tol = cfg.acc_tol_0

    subsolver_cfg = copy(cfg.subsolver_cfg)  # we will update this over time

    # sets up the augmented lagrangian function which will be minimized by the selected subsolver optimization method
    # prior to the update of the other parameters and the lagrangian multipliers

    augm_lagr = AugmentedLagrangian(f, gs, hs, list(*g_mults), list(*h_mults), penalty)
    for idx in range(cfg.max_iters):
        subsolver_cfg.min_step = acc_tol  # updates the expected accuracy inside the subsolver
        subsolver_result = cfg.subsolver_method(augm_lagr, p, mfld, cfg.subsolver_cfg, *args)
        if not subsolver_result.success:
            return RalmResult(
                success=False,
                p=subsolver_result.p,
                iters=idx + 1,
                history=RalmHistory(
                    p_hist=torch.tensor(p_hist),
                    f_hist=torch.tensor(f_hist),
                    gs_hist=torch.tensor(gs_hist),
                    hs_hist=torch.tensor(hs_hist),
                    acc_hist=torch.tensor(acc_hist),
                    penalty_hist=torch.tensor(penalty_hist),
                    g_mults_hist=torch.tensor(g_mults_hist),
                    h_mults_hist=torch.tensor(h_mults_hist),
                )
            )

        # assume that the subsolver was successful after this point
        next_p = subsolver_result.p
        if mfld.dist(p, next_p) < cfg.min_step and acc_tol < cfg.acc_tol_min:
            return RalmResult(
                success=True,
                p=next_p,
                iters=idx + 1,
                history=RalmHistory(
                    p_hist=torch.tensor(p_hist),
                    f_hist=torch.tensor(f_hist),
                    gs_hist=torch.tensor(gs_hist),
                    hs_hist=torch.tensor(hs_hist),
                    acc_hist=torch.tensor(acc_hist),
                    penalty_hist=torch.tensor(penalty_hist),
                    g_mults_hist=torch.tensor(g_mults_hist),
                    h_mults_hist=torch.tensor(h_mults_hist),
                )
            )

        # if convergence has not been achieved then we update the multipliers, etc.

        current_g_vals = [g.value(p, mfld, *args) for g in gs]
        current_h_vals = [h.value(p, mfld, *args) for h in hs]

        next_h_mults = torch.clamp(h_mults + penalty * torch.tensor(current_h_vals), cfg.h_mult_min, cfg.h_mult_max)
        next_g_mults = torch.clamp(g_mults + penalty * torch.tensor(current_g_vals), cfg.g_mult_min, cfg.g_mult_max)

        next_g_vals = [g.value(next_p, mfld, *args) for g in gs]
        next_sigma = torch.maximum(torch.tensor(next_g_vals), -g_mults / penalty)
        next_acc = max(cfg.acc_tol_min, cfg.acc_decay * acc_tol)

        next_h_vals = [h.value(next_p, mfld, *args) for h in hs]

        num_gs = len(gs)  # for clarity
        num_hs = len(hs)

        if idx == 0 or (
                # NOTE: relies on short-circuiting for sigma to not be None at this point
                torch.any(
                    torch.maximum(
                        torch.unsqueeze(torch.tensor(next_h_vals).abs(), -1).repeat((1, num_gs)),  # stacks horiz
                        torch.unsqueeze(next_sigma.abs(), 0).repeat((num_hs, 1))  # stacks vertically
                    ) <= cfg.ratio * torch.maximum(
                        torch.unsqueeze(torch.tensor(current_h_vals).abs(), -1).repeat((1, num_gs)),  # stacks horiz
                        torch.unsqueeze(sigma.abs(), 0).repeat((num_hs, 1))  # stacks vertically
                    )
                )
        ):
            next_penalty = penalty  # no change in penalty
        else:
            next_penalty = cfg.penalty_growth * penalty

        # update the values
        p = next_p

        g_mults = next_g_mults
        h_mults = next_h_mults

        sigma = next_sigma
        acc_tol = next_acc
        penalty = next_penalty

        # update the histories

        next_f_val = f.value(next_p, mfld, *args)

        p_hist.append(p)
        f_hist.append(next_f_val)
        gs_hist.append(next_g_vals)
        hs_hist.append(next_h_vals)
        acc_hist.append(next_acc)
        penalty_hist.append(penalty)
        g_mults_hist.append(g_mults)
        h_mults_hist.append(h_mults)

    # ran out of iterations without converging
    return RalmResult(
        success=False,
        p=p,
        iters=cfg.max_iters,
        history=RalmHistory(
            p_hist=torch.tensor(p_hist),
            f_hist=torch.tensor(f_hist),
            gs_hist=torch.tensor(gs_hist),
            hs_hist=torch.tensor(hs_hist),
            acc_hist=torch.tensor(acc_hist),
            penalty_hist=torch.tensor(penalty_hist),
            g_mults_hist=torch.tensor(g_mults_hist),
            h_mults_hist=torch.tensor(h_mults_hist),
        )
    )
