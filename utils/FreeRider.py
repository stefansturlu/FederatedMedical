from abc import ABCMeta
from torch import nn, Tensor, randn, tensor, long
from typing import Tuple


class FreeRider(metaclass=ABCMeta):
    @staticmethod
    def free_grads(model: nn.Module, prev_global: nn.Module) -> Tuple[Tensor, Tensor]:
        if prev_global is not None:
            # This is very hard to notice that it's a free-rider, need STD-DAGMM
            return FreeRider.delta_gradient_gen(model, prev_global)

        mean = 0
        std = 0

        for param in model.parameters():
            # We do the manipulation here for simplicity sake even if it makes more
            # sense to do it elsewhere due to grad values being None
            grad_m, grad_s = FreeRider.standard_gradient_gen(param)
            # grad_m, grad_s = FreeRider.delta_gradient_gen(param)
            mean += grad_m.mean()
            std += grad_s.std()

        return mean, std

    @staticmethod
    def normal_grads(model: nn.Module) -> Tuple[Tensor, Tensor]:
        mean = 0
        std = 0
        for param in model.parameters():
            mean += param.grad.mean()
            std += param.grad.std()

        return mean, std

    @staticmethod
    def standard_gradient_gen(param: nn.parameter.Parameter) -> Tuple[Tensor, Tensor]:
        R1 = 1e-4
        R2 = 10e-4
        grad_m = R1 * randn(param.data.size())
        grad_s = R2 * randn(param.data.size())

        return grad_m, grad_s

    @staticmethod
    def delta_gradient_gen(model: nn.Module, prev_global: nn.Module) -> Tuple[Tensor, Tensor]:
        dest = dict(prev_global.named_parameters())
        mean = 0
        std = 0

        for name, param in model.named_parameters():
            prev_param = dest[name]
            diff = param.data - prev_param.data

            mean += diff.mean()
            std += diff.std()

        return mean, std
