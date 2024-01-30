import torch

import pytorch_.integrate
from pytorch_.integrate import TSDFVolumeChild as TSDFVolumePytorch


class TSDFVolumeChild(TSDFVolumePytorch):

  using = "Using PyTorch JIT."

  def prepare(self):
    super().prepare()
    # same as adding @torch.jit.script
    pytorch_.integrate.integrate = torch.jit.script(pytorch_.integrate.integrate)