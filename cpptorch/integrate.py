import torch
import fusion_cpp

import pytorch_
from pytorch_.integrate import TSDFVolumeChild as TSDFVolumePytorch


class TSDFVolumeChild(TSDFVolumePytorch):
  using='Using PyTorch C++'

  def prepare(self):
    super().prepare()
    pytorch_.integrate.integrate = integrate
def integrate(
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