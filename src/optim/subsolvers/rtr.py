from typing import Callable

import torch
import numpy as np

from scipy.optimize import root_scalar

from dataclasses import dataclass, field

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import SubsolverCfg, CustomSubsolverResult, SubsolverResult, SubsolverHistory, SubsolverCriterion

REGION_RADIUS_EPS = 1E-6  # used to check equivalency of norm of eta updates
SUBPROBLEM_REL_ACC_EPS = 1E-6  # used to prevent nan in evaluating convergence criterion of eta updates

@dataclass
class RiemTrustRegionCfg(SubsolverCfg):
    radius_max: float
    radius_growth: float  # in (0, 1)
    radius_start: float # in (0, radius_max)
    min_quality_for_step: float  # in [0, 1/4)
    symm_lin_oper: Callable[[torch.Tensor], torch.Tensor] = field(default=lambda p: torch.eye(p.shape[0]))
    damp: float = 0.6
    criterion_mode: SubsolverCriterion
    criterion_eps: float = 0.01
    max_iters: int = 1000
    subproblem_max_iters = 10
    subproblem_rel_acc: float = 0.03  # 3 percent
    subproblem_damp: float = 0.6

@dataclass
class RiemTrustRegionHistory:
    p_hist: torch.Tensor
    f_hist: torch.Tensor
    quality_hist: torch.Tensor
    radius_hist: torch.Tensor

@dataclass
class RiemTrustRegionResult(CustomSubsolverResult):
    success: bool
    p: torch.Tensor
    iters: int
    history: RiemTrustRegionHistory

    @property
    def result(self) -> SubsolverResult:
        return SubsolverResult(
            success=self.success,
            p=self.p,
            iters=self.iters,
            history=SubsolverHistory(
                p_hist=self.history.p_hist,
                f_hist=self.history.f_hist,
            )
        )

def tr_subproblem_m(eta: torch.Tensor, p_k: torch.Tensor, f_k: torch.Tensor, grad_f_k: torch.Tensor, h_k: torch.Tensor, g_k: torch.Tensor) -> float:
    eta, p_k, f_k, grad_f_k, h_k, g_k = (eta.detach().numpy(), p_k.detach().numpy(), f_k.detach().numpy(), grad_f_k.detach().numpy(), h_k.detach().numpy(), g_k.detach().numpy())
    m_k = f_k + grad_f_k @ g_k @ eta + 0.5 * (h_k @ eta) @ g_k @ eta
    return m_k.item()

def _subproblem_grad(eta: np.ndarray, grad_f: np.ndarray, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    return g.T @ grad_f + h.T @ g @ eta

def _ellipsoid_cond_f(lam, d, y, b) -> float:
    result = 0.0
    for i in range(len(d)):
        result += d[i] * y[i]**2.0 / (1.0 + 2.0 * lam * d[i])**2.0
    result -= b

    return result

def _ellipsoid_cond_fprime(lam, d, y) -> float:
    result = 0.0
    for i in range(len(d)):
        result += d[i]**2.0 * y[i]**2.0 / (1.0 + 2.0 * lam * d[i])**3.0
    result *= -4.0

    return result

def _project_onto_ellipsoid(x: np.ndarray, p: np.ndarray, b: float) -> np.ndarray:
    # if the point is already in the ellipsoid then we don't need to perform a projection
    if x @ p @ x <= b:
        return x
    else:
        d, q = np.linalg.eigh(p)
        y = q @ x

        result = root_scalar(f=lambda soln_lam: _ellipsoid_cond_f(soln_lam, d, y, b), x0=1.0, fprime=lambda soln_lam: _ellipsoid_cond_fprime(soln_lam, d, y))
        lam = result.root

        proj_x = np.linalg.inv(np.eye(len(d)) + 2.0*lam*p) @ x
        return proj_x

def solve_tr_subproblem_m(p_k: torch.Tensor, grad_f_k: torch.Tensor, h_k: torch.Tensor, g_k: torch.Tensor, radius_k: float, max_iters: int, rel_acc: float, damp: float) -> torch.Tensor:
    # implement proximal mapping to solve subproblem

    # NOTE: converts to numpy to speed up computation (and not worry about torch computational graph)
    p_k: np.ndarray = p_k.detach().numpy()
    grad_f_k: np.ndarray = grad_f_k.detach().numpy()
    h_k: np.ndarray = h_k.detach().numpy()
    g_k: np.ndarray = g_k.detach().numpy()

    eta = np.zeros(p_k)
    for idx in range(max_iters):
        # implements proximal updates
        m_grad = _subproblem_grad(eta, grad_f_k, h_k, g_k)
        eta_upd = _project_onto_ellipsoid(eta - damp * m_grad, g_k, radius_k**2.0)

        # exits early if the relative accuracy of any component is below the required threshold
        rel_err = np.abs(eta_upd - eta) / (np.abs(eta) + SUBPROBLEM_REL_ACC_EPS * np.ones_like(eta))
        if np.max(rel_err) <= rel_acc:
            eta = eta_upd
            break

        eta = eta_upd

    # only need approximate solution so no guarantees that all the iterations completed successfully
    return torch.tensor(eta)

def riem_trust_region(
    f: MfldFunc,
    p0: torch.Tensor,
    mfld: ComputeMfld,
    cfg: RiemTrustRegionCfg,
    *args: *FuncArgs) -> RiemTrustRegionResult:

    p_prev = None  # track this value to utilize convergence criterion
    p: torch.Tensor = p0.detach().clone()  # cloned so can be modified without changing p_prev

    radius = cfg.radius_start

    p_hist = []
    f_hist = []
    quality_hist = []
    radius_hist = []

    for idx in range(cfg.max_iters):
        metric = mfld.mfld.metric(p)

        f_curr: torch.Tensor = f.value(p, mfld, *args)
        f_diff_curr: torch.Tensor = f.diff(p, mfld, *args)
        f_grad_curr: torch.Tensor = metric.sharp(f_diff_curr) # gets gradient using the metric
        h_curr: torch.Tensor = cfg.symm_lin_oper(p)
        g_curr: torch.Tensor = metric.mat  # metric matrix

        # approximately solve the trust-region subproblem
        curr_eta = solve_tr_subproblem_m(p, f_grad_curr, h_curr, g_curr, radius, cfg.subproblem_max_iters, cfg.subproblem_rel_acc, cfg.subproblem_damp)
        p_possible_upd = mfld.exp(p, -cfg.damp * curr_eta)  # predicted next point under retraction

        quality_curr = (f.value(p, mfld, *args) - f.value(p_possible_upd, mfld, *args)) / (tr_subproblem_m(torch.zeros_like(curr_eta), p, f_curr, f_grad_curr, h_curr, g_curr) - tr_subproblem_m(curr_eta, p, f_curr, f_grad_curr, h_curr, g_curr))

        if quality_curr < 0.25:
            radius *= 0.25
        elif quality_curr > 0.75 and torch.abs(torch.linalg.norm(curr_eta) - radius) <= REGION_RADIUS_EPS:
            radius = min(2.0 * radius, cfg.radius_max)

        if quality_curr > cfg.min_quality_for_step:
            p = p_possible_upd

        # update histories
        p_hist.append(p)
        f_hist.append(f.value(p, mfld, *args).item())
        quality_hist.append(quality_curr)
        radius_hist.append(radius)

        if p_prev is not None:
            if cfg.criterion_mode == SubsolverCriterion.DISTANCE:
                dist = mfld.dist(p_prev, p)
                if dist <= cfg.criterion_eps
                    return RiemTrustRegionResult(
                        success=True,
                        p=p,
                        iters=idx + 1,
                        history=RiemTrustRegionHistory(
                            p_hist=torch.tensor(p_hist),
                            f_hist=torch.tensor(f_hist),
                            quality_hist=torch.tensor(quality_hist),
                            radius_hist=torch.tensor(radius_hist),
                        )
                    )
            elif cfg.criterion_mode == SubsolverCriterion.NORM:
                # computes the norm of the gradient of the function
                f_diff = f.diff(p, mfld, *args)
                metric = mfld.mfld.metric(p)
                f_grad = metric.sharp(f_diff)
                f_grad_norm = metric(f_grad, f_grad)

                if f_grad_norm <= cfg.criterion_eps:
                    return RiemTrustRegionResult(
                        success=True,
                        p=p,
                        iters=idx + 1,
                        history=RiemTrustRegionHistory(
                            p_hist=torch.tensor(p_hist),
                            f_hist=torch.tensor(f_hist),
                            quality_hist=torch.tensor(quality_hist),
                            radius_hist=torch.tensor(radius_hist),
                        )
                    )


        p_prev = p.clone()
    return RiemTrustRegionResult(
        success=False,
        p=p,
        iters=cfg.max_iters,
        history=RiemTrustRegionHistory(
            p_hist=torch.tensor(p_hist),
            f_hist=torch.tensor(f_hist),
            quality_hist=torch.tensor(quality_hist),
            radius_hist=torch.tensor(radius_hist),
        )
    )