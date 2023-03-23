from torch import nn
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d, MaxPool3d
from onnx2pytorch.operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (MaxPool1d, MaxPool2d, MaxPool3d, Loop, LSTMWrapper, Split, TopK)
STANDARD_LAYERS = (
    Conv1d, Conv2d, Conv3d,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)

