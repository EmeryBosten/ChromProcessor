from typing import List, Tuple
from numpy.typing import NDArray

import numpy as np


class PeakModel:
    """Abstract class for functions that model peak shape"""

    def __call__(self, t: float | NDArray, *params: float) -> float | NDArray:
        """
        Returns height of peak specified by params at given time `t`.
        """
        return self.val(t, *params)

    def val(self, t: float | NDArray, *params: float) -> float | NDArray:
        """
        Returns height of peak specified by params at given time `t`.
        """
        raise NotImplemented

    def grad(self, t: float | NDArray, *params: float) -> NDArray:
        """
        Returns gradient w.r.t params. If t is array, the return shape is [parameter, t]
        """
        raise NotImplemented

    def init_guess(
        self, height: float, maximum: float, width_left: float, width_right: float
    ) -> NDArray:
        """
        Creates initial guess of the model parameters from peak descriptors (height, location of maximum, left and right widths of peak)
        """
        raise NotImplemented

    def get_bounds(self, max_t: float) -> List[Tuple[float, float]]:
        """
        Returns bounds for individual parameters in the shape [(p1 min, p1 max), (p2 min, p2 max), ...].

        `max_t` is the maximum `t` that will be passed into the model, and also the number of time points in the peak region.
        """
        raise NotImplemented

    def n_params(self) -> int:
        """Returns number of parameters of the model"""
        raise NotImplemented


def test_gradients(model: PeakModel, t: NDArray, params: List[float]) -> bool:
    """
    Compares gradient from the model with numerical gradients. Returns True is all gradients are close.

    This is useful when making sure that gradients in new peak model are all correct.
    """
    eps = 1e-5
    grad = model.grad(t, *params)

    for pidx, pgrad in enumerate(grad):
        pl = params.copy()
        pr = params.copy()
        pl[pidx] -= eps
        pr[pidx] += eps
        num_grad = (model.val(t, *pr) - model.val(t, *pl)) / (2 * eps)

        if not np.allclose(pgrad, num_grad, rtol=1e-5):
            print("Gradient does not match for parameter", pidx + 1)
            print(pgrad)
            print(num_grad)
            return False
    return True

