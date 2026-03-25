import torch

from enum import Enum

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import CustomSubsolverResult, SubsolverResult, SubsolverHistory, SubsolverCfg
from optim.subsolvers.rgd import riem_grad_descent, RiemGradDescentCfg


class SubsolverMethod(Enum):
    RIEM_GRAD_DESCENT = (riem_grad_descent, RiemGradDescentCfg)

    def __call__(self, f: MfldFunc, p0: torch.Tensor, mfld: ComputeMfld, cfg: SubsolverCfg,
                 *args: *FuncArgs) -> SubsolverResult:
        method, cfg_type = self.value

        if not isinstance(cfg, cfg_type):
            raise ValueError(f"configuration of type {type(cfg)} cannot be used with subsolver method {self.name}")

        result = method(f, p0, mfld, cfg, *args)  # calls the method (with having checked that the correct cfg is given)
        return result.result  # converts from custom return type to the main return type
