import torch

from dataclasses import dataclass

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import SubsolverCfg

from src.optim.results import CustomSubsolverResult, SubsolverHistory, SubsolverResult


@dataclass
class RiemGradDescentCfg(SubsolverCfg):
    damp: float = 0.6
    min_step: float = 0.01
    max_iters: int = 1000


@dataclass
class RiemGradDescentHistory:
    p_hist: torch.Tensor
    f_hist: torch.Tensor


@dataclass
class RiemGradDescentResult(CustomSubsolverResult):
    success: bool
    p: torch.Tensor
    iters: int
    history: RiemGradDescentHistory

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


def riem_grad_descent(f: MfldFunc, p0: torch.Tensor, mfld: ComputeMfld, cfg: RiemGradDescentCfg, *args: *FuncArgs):
    p_prev = None  # track this value to utilize convergence criterions
    p: torch.Tensor = p0.detach().clone()  # cloned so can be modified without changing p_prev (when assigned)

    p_hist = []
    f_hist = []

    for idx in range(cfg.max_iters):
        df = f.diff(p, mfld, *args)  # cotangent space
        grad_f = mfld.mfld.metric(p).sharp(df)  # tangent space

        p = mfld.exp(p, -cfg.damp * grad_f)

        # update the histories
        p_hist.append(p)
        f_hist.append(f.value(p, mfld, *args).item())

        if p_prev is not None:
            dist = mfld.dist(p_prev, p)
            if dist < cfg.min_step:
                return RiemGradDescentResult(
                    success=True,
                    p=p,
                    iters=idx + 1,
                    history=RiemGradDescentHistory(
                        p_hist=torch.tensor(p_hist),
                        f_hist=torch.tensor(f_hist),
                    ))

        p_prev = p.clone()  # otherwise if we modify p then p_prev is changed
    return RiemGradDescentResult(
        success=False,
        p=p,
        iters=cfg.max_iters,
        history=RiemGradDescentHistory(
            p_hist=torch.tensor(p_hist),
            f_hist=torch.tensor(f_hist),
        ))
