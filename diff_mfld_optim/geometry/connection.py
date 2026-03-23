import torch


from torch.func import jacrev

from abc import abstractmethod, ABC


class Connection(ABC):
    def __init__(self, n, r):
        self._n = n  # dimension of the underlying space
        self._r = r  # dimension of the bundle vector space

    @property
    def n(self) -> int:
        return self._n

    @property
    def r(self) -> int:
        return self._r

    @abstractmethod
    def _eval(self, p) -> torch.Tensor:
        pass

    def __call__(self, *coords_sets) -> torch.Tensor:
        # in the case of product manifolds then we merge the coordinates
        p = torch.cat(tuple(coords for coords in coords_sets))
        coords_n = p.shape[0]
        if coords_n != self.n:
            raise ValueError(
                "Coordinates passed to connection does not match dimension of "
                f"underlying manifold: n={self.n}, coords_n={coords_n}"
            )

        return self._eval(p)

    def partials(self, p: torch.Tensor, order: int = 1) -> torch.Tensor:
        if order < 1:
            raise ValueError(
                "derivative order of connection coefficient partials must be greater than 0"
            )
        
        fn = jacrev(lambda p: self._eval(p))
        for _ in range(1, order):
            fn = jacrev(fn)

        components = fn(p)
        return components
