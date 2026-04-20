#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class Gaussian_loss(_Loss):

    __constants__ = ["full", "eps", "reduction"]
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor | float) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.gaussian_nll_loss(
            input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )
    
    # __constants__ = ["full", "eps", "reduction"]
    # full: bool
    # eps: float
    # reduction: str
    
    # def __init__(self, *, full: bool = False, eps: float = 1e-6,
    #              reduction: str = "mean", size_average=None,
    #              reduce=None) -> None:
        
    #     if size_average is not None or reduce is not None:
    #         self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
    #     else:
    #         self.reduction = reduction
            
    #     self.full = full
    #     self.eps = eps

        
    # def forward(self, input: Tensor, target: Tensor, var: Tensor | float) -> Tensor:
        
    #     ##  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html

    #     ##  The reduction is how each (x, y) sample's loss function
    #     ##  output is handled, when combined.
    #     ##  'mean' removes dependency on ratios within dataset, by
    #     ##  taking mean of loss function output (unlike 'sum').

    #     ##  Input regards is the prediction made by the model, based
    #     ##  on the input provided.

    #     ##  Target is the dataset's target, mean of the distribution,
    #     ##  can be initialised randomly, then iterated.

    #     ##  Var, distribution variance, can be initialised randomly,
    #     ##  and then iterated.
        
    #     return F.gaussian_nll_loss(
    #         input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
    #     )

