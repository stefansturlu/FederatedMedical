from utils.typings import FreeRiderAttack
from torch import nn, Tensor, randn, tensor, device, float64
from typing import Tuple


class FreeRider:
    def __init__(
        self,
        device: device,
        attack: FreeRiderAttack
    ):
        self.device = device
        self.attack = attack

    def free_grads(self, model: nn.Module, prev_global: nn.Module) -> Tuple[Tensor, Tensor]:
        mean = tensor(0, device=self.device, dtype=float64)
        std = tensor(0, device=self.device, dtype=float64)

        if self.attack == FreeRiderAttack.BASIC:
            return mean, std


        if self.attack == FreeRiderAttack.DELTA and prev_global is not None:
            # This is very hard to notice that it's a free-rider, need STD-DAGMM or privacy amplification
            return self.delta_gradient_gen(model, prev_global)


        if self.attack == FreeRiderAttack.NOISY or self.attack == FreeRiderAttack.DELTA:

            for param in model.parameters():
                # We do the manipulation here for simplicity sake even if it makes more
                # sense to do it elsewhere due to grad values being None
                grad_m, grad_s = self.standard_gradient_gen(param)
                # grad_m, grad_s = self.delta_gradient_gen(param)
                mean += grad_m.mean()
                std += grad_s.std()

        return mean, std


    def normal_grads(self, model: nn.Module) -> Tuple[Tensor, Tensor]:
        mean = tensor(0, device=self.device, dtype=float64)
        std = tensor(0, device=self.device, dtype=float64)
        for param in model.parameters():
            mean += param.grad.mean()
            std += param.grad.std()

        return mean, std


    def standard_gradient_gen(self, param: nn.parameter.Parameter) -> Tuple[Tensor, Tensor]:
        R1 = 1e-4
        R2 = 10e-4
        grad_m = R1 * randn(param.data.size())
        grad_s = R2 * randn(param.data.size())

        return grad_m, grad_s


    def delta_gradient_gen(self, model: nn.Module, prev_global: nn.Module) -> Tuple[Tensor, Tensor]:
        dest = dict(prev_global.named_parameters())
        mean = tensor(0, device=self.device, dtype=float64)
        std = tensor(0, device=self.device, dtype=float64)

        for name, param in model.named_parameters():
            prev_param = dest[name]
            diff = param.data - prev_param.data

            mean += diff.mean()
            std += diff.std()

        return mean, std
