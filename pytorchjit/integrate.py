import torch

from pytorch_.integrate import TSDFVolumePytorch


class TSDFVolumePytorchJIT(TSDFVolumePytorch):

  using = "Using PyTorch JIT."

  def prepare(self):
    super().prepare()
    # same as adding @torch.jit.script
    self.integrate_static = torch.jit.script(self.integrate_static)