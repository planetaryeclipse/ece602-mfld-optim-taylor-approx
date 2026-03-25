import itertools
import numpy as np
import torch

import warnings
import inspect

from enum import Enum
from scipy.optimize import root

from src.diff_mfld.geometry.metric import RnMetricField
from src.diff_mfld.geometry.connection import Connection

from typing import List, Callable


# wrapper class which matches the expected signature and then computes all required connection coefficients (and
# partials if applicable) before passing it to the approximate methods below

# NOTE: cannot specify callable arguments as [np.ndarray, np.ndarray, ...] so I just use the full variadic below, it is
# assumed that there are always 2 occupied ndarray arguments and the remainder are the connection coefficients with
# increasing derivative order of partials


def _count_conn_coeff_args(map):
    sig = inspect.signature(map)
    coeffs_count = len(sig.parameters) - 2

    return coeffs_count


def _compute_conn_coeff_args(
        count: int, p: torch.Tensor, conn: Connection
) -> List[np.ndarray]:
    conn_coeffs_with_partials = []
    if count > 0:
        conn_coeffs_with_partials.append(conn(p).detach().numpy())
    for deriv_order in range(1, count):
        conn_coeffs_with_partials.append(conn.partials(p, deriv_order).detach().numpy())

    return conn_coeffs_with_partials


class ApproxExpMapWrapper:
    def __init__(self, exp_map: Callable[..., np.ndarray]):
        self._exp_map = exp_map
        self._coeffs_count = _count_conn_coeff_args(exp_map)

    def __call__(
            self, p: torch.Tensor, v: torch.Tensor, conn: Connection
    ) -> torch.Tensor:
        conn_coeffs_with_partials = _compute_conn_coeff_args(
            self._coeffs_count, p, conn
        )

        p_numpy = p.detach().numpy()
        v_numpy = v.detach().numpy()

        q_numpy = self._exp_map(p_numpy, v_numpy, *conn_coeffs_with_partials)
        return torch.tensor(q_numpy, dtype=p.dtype)


class ApproxLogMapWrapper:
    def __init__(self, log_map: Callable[..., np.ndarray]):
        self._log_map = log_map
        self._coeffs_count = _count_conn_coeff_args(log_map)

    def __call__(
            self, p: torch.Tensor, q: torch.Tensor, conn: Connection
    ) -> torch.Tensor:
        conn_coeffs_with_partials = _compute_conn_coeff_args(
            self._coeffs_count, p, conn)

        p_numpy = p.detach().numpy()
        q_numpy = q.detach().numpy()

        v_numpy = self._log_map(p_numpy, q_numpy, *conn_coeffs_with_partials)
        return torch.tensor(v_numpy, dtype=p.dtype)


# approximated exponential and logarithmic maps

# NOTE: the following can technically be extended to computing values along the geodesic in a generic manner but we're
# explicitly handling the exponential adn logarithmic cases where t=1 so we just omit the (t-0)^k term in the Taylow
# expansions in the functions below


def approx_exp_map_o1(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    f0_val = f0(v)
    return p + v


def approx_exp_map_o2(
        p: np.ndarray, v: np.ndarray, conn_coeffs: np.ndarray
) -> np.ndarray:
    f0_val = f0(v)
    f1_val = f1(v, conn_coeffs)
    return p + f0_val + f1_val


def approx_exp_map_o3(
        p: np.ndarray,
        v: np.ndarray,
        conn_coeffs: np.ndarray,
        conn_coeffs_partials: np.ndarray,
) -> np.ndarray:
    f0_val = f0(v)
    f1_val = f1(v, conn_coeffs)
    f2_val = f2(v, conn_coeffs, conn_coeffs_partials)
    return p + f0_val + f1_val + f2_val


def approx_log_map_o1(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return q - p  # no need to use solver here


def _approx_log_map(f_fn, fprime_fn, p, q, order):
    v_guess = approx_log_map_o1(p, q)
    result = root(f_fn, v_guess, jac=fprime_fn)

    if result.success:
        v = result.y[:, 0]
    else:
        warnings.warn(
            f"failed to find solution to order {order} approx log map, falling back to order 1 estimate"
        )
        v = v_guess
    return v.astype(dtype=p.dtype)  # always returns float64 for some reason


def approx_log_map_o2(
        p: np.ndarray, q: np.ndarray, conn_coeffs: np.ndarray
) -> np.ndarray:
    f_fn = lambda v: -q + (p + f0(v) + f1(v, conn_coeffs))
    fprime_fn = lambda v: f0_diff_wrt_y(v) + f1_diff_wrt_y(v, conn_coeffs)

    return _approx_log_map(f_fn, fprime_fn, p, q)


def approx_log_map_o3(
        p: np.ndarray,
        q: np.ndarray,
        conn_coeffs: np.ndarray,
        conn_coeffs_partials: np.ndarray,
) -> np.ndarray:
    f_fn = lambda v: -q + (
            p + f0(v) + f1(v, conn_coeffs) + f2(v, conn_coeffs, conn_coeffs_partials)
    )
    fprime_fn = (
        lambda v: f0_diff_wrt_y(v)
                  + f1_diff_wrt_y(v, conn_coeffs)
                  + f2_diff_wrt_y(v, conn_coeffs, conn_coeffs_partials)
    )

    return _approx_log_map(f_fn, fprime_fn, p, q)


# implements the higher order component calculations


def f0(y: np.ndarray) -> np.ndarray:
    return y


def f0_diff_wrt_y(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    return np.eye(n)


def f1(y: np.ndarray, conn_coeffs: np.ndarray) -> np.ndarray:
    return -torch.tensordot(torch.tensordot(conn_coeffs, y, ([2], [0])), y, ([1], [0]))


def f1_diff_wrt_y(y: np.ndarray, conn_coeffs: np.ndarray) -> np.ndarray:
    return -torch.tensordot(conn_coeffs, y, ([1], [0]))


def f2(
        y: np.ndarray, conn_coeffs: np.ndarray, conn_coeffs_partials: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(y)

    result += -torch.tensordot(
        torch.tensordot(
            torch.tensordot(conn_coeffs_partials, y, ([3], [0])), y, ([2], [0])
        ),
        y,
        ([1], [0]),
    )
    result += torch.tensordot(
        torch.tensordot(
            torch.tensordot(
                torch.tensordot(conn_coeffs, y, ([1], [0])), conn_coeffs, ([1], [0])
            ),
            y,
            ([2], [0]),
        ),
        y,
        ([1], [0]),
    )
    result += torch.tensordot(
        torch.tensordot(
            torch.tensordot(
                torch.tensordot(conn_coeffs, conn_coeffs, ([1], [0])), y, ([3], [0])
            ),
            y,
            ([2], [0]),
        ),
        y,
        ([1], [0]),
    )
    return result


def f2_diff_wrt_y(
        y: np.ndarray, conn_coeffs: np.ndarray, conn_coeffs_partials: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(y)

    result += -torch.tensordot(
        torch.tensordot(conn_coeffs_partials, y, ([3], [0])), y, ([2], [0])
    )
    result += (
        torch.tensordot(
            torch.tensordot(
                torch.tensordot(conn_coeffs, y, ([1], [0])), conn_coeffs, ([1], [0])
            ),
            y,
            ([2], [0]),
        ),
    )
    result += torch.tensordot(
        torch.tensordot(
            torch.tensordot(conn_coeffs, conn_coeffs, ([1], [0])), y, ([3], [0])
        ),
        y,
        ([2], [0]),
    )
    return result

# older approximate code from research paper (now unused)

# def _inspect_conn_coeffs(conn_coeffs: np.ndarray) -> Tuple[int, int]:
#     n = conn_coeffs.shape[0]
#     r = conn_coeffs.shape[2]

#     if n != conn_coeffs.shape[1]:
#         raise ValueError(
#             f"connection coefficients must share length on dims 0, 1: dim0={n}, dim1={conn_coeffs.shape[1]}"
#         )

#     return n, r


# def _solve_geod_pos_so(
#     p: np.ndarray, v: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
# ) -> np.ndarray:
#     # solves for the updated position along the geodesic using the second order
#     # approximation of the geodesic (requires a tangent bundle connection)

#     n, r = _inspect_conn_coeffs(conn_coeffs)
#     if n != r:
#         raise ValueError(
#             f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
#         )

#     q = np.zeros((n), dtype=p.dtype)  # otherwise causes issues in the chain
#     for k in range(n):
#         q[k] += p[k] + v[k] * (t - t0)
#         for i, j in itertools.product(range(n), range(n)):
#             q[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2

#     return q


# def _initial_geod_vel_f_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
#     n = len(v)

#     f = np.zeros(n, dtype=p.dtype)
#     for k in range(n):
#         f[k] = v[k] * (t - t0) + (p[k] - q[k])
#         for i, j in itertools.product(range(n), range(n)):
#             f[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2
#     return f


# def _initial_geod_vel_fprime_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
#     n = len(v)

#     df_dv = np.zeros((n, n), dtype=p.dtype)
#     for k, i in itertools.product(range(n), range(n)):
#         if k == i:
#             df_dv[k] += t - t0
#         for j in range(n):
#             df_dv[k, i] += (
#                 -0.5
#                 * (conn_coeffs[k, i, j] + conn_coeffs[k, j, i])
#                 * v[j]
#                 * (t - t0) ** 2
#             )
#     return df_dv


# def _solve_initial_geod_vel_so(
#     p: np.ndarray, q: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
# ) -> np.ndarray:
#     # solves for the initial velocity along the geodesic using the second order
#     # approximation of the geodesic (requires a tangent bundle connection)

#     n, r = _inspect_conn_coeffs(conn_coeffs)
#     if n != r:
#         raise ValueError(
#             f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
#         )

#     v_guess = (q - p) / (t - t0)  # Euclidean is the guess

#     result = root(
#         _initial_geod_vel_f_so,
#         v_guess,
#         args=(p, q, conn_coeffs, t, t0),
#         jac=_initial_geod_vel_fprime_so,
#     )

#     if result.success:
#         # root operation computes a float64 for some reason
#         return result.x.astype(dtype=p.dtype)
#     else:
#         warnings.warn(
#             "failed to find solution to logarithmic map given conditions "
#             f"p={p}, q={q}, t0={t0}, t={t}, conn_coeffs={conn_coeffs}, "
#             "falling back to Euclidean estimate"
#         )
#         return v_guess
