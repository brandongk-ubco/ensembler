from enum import Enum
from .Best import Best


class EnsembleMethods(Enum):
    best = "best"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(method):
        if method == "best":
            return Best

        raise ValueError("Ensemble Method {} not defined".format(method))


__all__ = [EnsembleMethods]
