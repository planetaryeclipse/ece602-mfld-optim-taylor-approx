import torch


# the various metrics describing the curvature of the surface of optimization

def euclid_metric(x1, x2, x3, scaling: float):
    metric = torch.eye(3)

    # pulls the metric "back" onto the scaled coordinate system
    factor = 1. / scaling ** 2
    metric = factor * metric  # avoids in-place operation to keep pytorch gradients

    return metric


def scaled_metric(x1, x2, x3, scaling: float):
    # elements are assigned to this metric directly to preserve gradient history
    metric = torch.zeros((3, 3))
    metric[0, 0] = (scaling * x1) ** 2 + 1.  # constant prevents degeneracy at origin
    metric[1, 1] = (scaling * x2) ** 2 + 1.
    metric[2, 2] = (scaling * x3) ** 2 + 1.

    # pulls the metric "back" onto the scaled coordinate system
    factor = 1. / scaling ** 2
    metric = factor * metric  # avoids in-place operation to keep pytorch gradients

    return metric


def coupled_metric(x1, x2, x3, scaling: float):
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
    metric = factor * metric  # avoids in-place operation to keep pytorch gradients

    return metric


def asymmetric_metric(x1, x2, x3, scaling: float):
    metric = torch.zeros((3, 3))
    metric[0, 0] = (scaling * x1) ** 2 + 1.
    metric[1, 1] = 0.5 * (x1 * scaling) ** 2 * (x2 * scaling) ** 2 + 1.
    metric[2, 2] = 0.25 * (x1 * scaling) ** 2 * (x2 * scaling) ** 2 * (x3 * scaling) ** 2 + 1.

    # pulls the metric "back" onto the scaled coordinate system
    factor = 1. / scaling ** 2
    metric = factor * metric  # avoids in-place operation to keep pytorch gradients

    return metric
