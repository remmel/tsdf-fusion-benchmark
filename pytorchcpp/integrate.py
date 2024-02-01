import torch
import fusion_cpp

import pytorch_
from pytorch_.integrate import TSDFVolumePytorch


class TSDFVolumePyTorchCpp(TSDFVolumePytorch):
  using='Using PyTorch C++'
  useGpu = None #Can handle both with and without GPU

  def prepare(self):
    super().prepare()

  @staticmethod
  def integrate_static(
          color_im,
          depth_im,
          cam_intr,
          cam_pose,
          obs_weight,
          world_c,
          vox_coords,
          weight_vol,
          tsdf_vol,
          color_vol,
          sdf_trunc,
          im_h,
          im_w):
    weight_vol, tsdf_vol, color_vol = fusion_cpp.integrate(
      world_c,
      vox_coords,
      weight_vol,
      tsdf_vol,
      color_vol,
      color_im,
      depth_im,
      cam_intr,
      cam_pose,
      im_h,
      im_w,
      sdf_trunc,
      obs_weight,
    )
    return weight_vol, tsdf_vol, color_vol
