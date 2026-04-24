import torch
import numpy as np

from diff_mfld.geodesic.geodesic_funcs import LogMethod, ExpMethod
from diff_mfld.geometry.metric import RnMetricField, MetricField
from diff_mfld.mfld import ComputeMfld, Mfld
from optim.subsolvers.rtr import RiemTrustRegionCfg, riem_trust_region
from src.optim.subsolvers.rgd import (
    riem_grad_descent,
    RiemGradDescentCfg,
    RiemGradDescentHistory,
    RiemGradDescentResult
)

from problem import create_problem, TargetCost

scaling = 1.5

target = torch.tensor([2.0, 3.0, 10.0])
target /= np.linalg.norm(target)
target *= scaling

cost_fn, _ = create_problem(target)

print(type(cost_fn))
print(cost_fn.target)

def coupled_metric(x1, x2, x3, scaling: float):
    factor = 1. / scaling ** 2

    metric = torch.zeros((3, 3))
    metric[0, 0] = (scaling * x1) ** 2 + 1.
    metric[0, 1] = 0.5 * (scaling * x1) * (scaling * x2)
    metric[0, 2] = 0.5 * (scaling * x1) * (scaling * x3)

    metric[1, 0] = 0.5 * (scaling * x2) * (scaling * x1)
    metric[1, 1] = (scaling * x2) ** 2 + 1.
    metric[1, 2] = 0.5 * (scaling * x2) * (scaling * x3)

    metric[2, 0] = 0.5 * (scaling * x3) * (scaling * x1)
    metric[2, 1] = 0.5 * (scaling * x3) * (scaling * x2)
    metric[2, 2] = (scaling * x3) ** 2 + 1.

    # pulls the metric "back" onto the scaled coordinate system
    factor = 1. / scaling ** 2
    metric = factor * metric # avoids in-place operation to keep pytorch gradients

    return metric

# sets up the manifold space
# metric = MetricField(lambda x1, x2, x3: coupled_metric(x1, x2, x3, scaling=scaling ))# scaling))
metric = MetricField(lambda x1, x2, x3: 1./scaling**2 * torch.eye(3), 3)
conn = metric.christoffels()
mfld = Mfld(metric, conn)
compute_mfld = ComputeMfld(mfld)

compute_mfld.exp_method = ExpMethod.APPROX_O3
compute_mfld.log_method = LogMethod.APPROX_O3
compute_mfld.dist_method = LogMethod.APPROX_O3


# sets up and runs optimization
cfg = RiemTrustRegionCfg()

p0 = torch.tensor([-5., -4., 3.])
p0 /= np.linalg.norm(p0)
p0 *= scaling

result = riem_trust_region(cost_fn, p0, compute_mfld, cfg, ())
print(f"success: {result.success}")

print("p_hist:")
print(result.history.p_hist)
print()
print("f_hist:")
print(result.history.f_hist)

print(f"Euclidean distance: {(np.linalg.norm(result.p - target) / scaling)}")
