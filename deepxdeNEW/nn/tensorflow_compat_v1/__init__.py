"""Package for tensorflow.compat.v1 NN modules."""
from __future__ import absolute_import

from .bionet import BiONet
from .deeponet import DeepONet, DeepONetCartesianProd, FourierDeepONetCartesianProd
from .fnn import FNN, PFNN
from .mfnn import MfNN
from .mfonet import MfONet
from .msffn import MsFFN, STMsFFN
from .nn import NN
from .resnet import ResNet

from .mfnn_L2H import MfNN_L2H   ##### Add for the hierarchical multi-fidelity NN, L2H
from .mfnn_LH2 import MfNN_LH2   ##### Add for the hierarchical multi-fidelity NN, LH2
from .mfnn_LMH import MfNN_LMH   ##### Add for the hierarchical multi-fidelity NN, LMH


__all__ = [
    "BiONet",
    "DeepONet",
    "DeepONetCartesianProd",
    "FourierDeepONetCartesianProd",
    "FNN",
    "PFNN",
    "MfNN",
    "MfONet",
    "MsFFN",
    "NN",
    "STMsFFN",
    "ResNet",
    "MfNN_L2H",
    "MfNN_LH2",
    "MfNN_LMH",
]
