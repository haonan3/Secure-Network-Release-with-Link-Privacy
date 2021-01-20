
"""Functional interface"""

import warnings
import math
from operator import mul
from functools import reduce
import sys

import torch
#from torch._C import _infer_size, _add_docstr
#from . import _functions
from torch.nn import _functions
#from .modules import utils
from torch.nn.modules import utils
#from ._functions.linear import Bilinear
#from torch.nn._functions.linear import Bilinear
#from ._functions.padding import ConstantPadNd
#from torch.nn._functions.padding import ConstantPadNd
#from ._functions import vision
#from torch.nn._functions import vision
#from ._functions.thnn.fold import Col2Im, Im2Col
#from torch.nn._functions.thnn.fold import Col2Im,Im2Col
from torch.autograd import Variable
#from .modules.utils import _single, _pair, _triple
#from torch.nn.modules.utils import _single, _pair, _triple


'''
Linear layer modified for PX gradients

The code is due to Mikko Heikkilä (@mixheikk)
'''


# Note: bias not checked yet
def linear(input, weight, bias=None, batch_size=None, for_test=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        if batch_size is None:
          return torch.mm(input, weight.t())
        else:
          print('fused op in functional.linear not implemented yet!')
          sys.exit(1)
          return torch.addmm(bias, input, weight.t())

    if for_test:
        if len(list(input.shape)) == 3:
            input = input.view(input.shape[0], input.shape[2])
        # output = input.matmul(torch.transpose(weight,-2,-1)[0])
        output = torch.mm(input, weight[0].t())
        assert len(list(output.shape)) == 2
    else:
        # output = input.matmul(torch.transpose(weight,-2,-1))
        output = torch.bmm(input, weight.permute(0,2,1))
        output = output.view(output.shape[0], output.shape[2])
        assert len(list(output.shape)) == 2

    # kts bias kun muu toimii
    if bias is not None:
        output += bias
    return output
