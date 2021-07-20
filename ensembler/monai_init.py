from monai.networks.layers.factories import Act
from ensembler.activations import Cos, LiHT, PWLinear
from torch.nn import Identity


@Act.factory_function("cos")
def cos_factory():
    return Cos


@Act.factory_function("liht")
def liht_factory():
    return LiHT


@Act.factory_function("linear")
def linear_factory():
    return Identity


@Act.factory_function("piecewiselinear")
def linear_factory():
    return PWLinear


def initialize():
    assert Act.COS == 'COS'
    assert Act.LIHT == 'LIHT'
    assert Act.LINEAR == 'LINEAR'
    assert Act.PIECEWISELINEAR == 'PIECEWISELINEAR'
