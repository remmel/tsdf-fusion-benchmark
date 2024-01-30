import torch

from fusion import TSDFVolume


class TSDFVolumeChild(TSDFVolume):
  using = "Using vanilla PyTorch."

  def __init__(self, vol_bnds, voxel_size, useGPU = True):
    self.useGPU = useGPU
    if (useGPU):
      if (not torch.cuda.is_available()):
        raise "No GPU detected."
      self.device = torch.device("cuda")
      print("Use GPU")
    else:
      self.device = torch.device("cpu")
      print("Use CPU")

    super().__init__(vol_bnds, voxel_size)


  def prepare(self):
    self.initVoxCoords()
    self.initWorldCoords()

    self._world_c = torch.from_numpy(self._world_c).float().to(self.device)
    self._vox_coords = torch.from_numpy(self._vox_coords).to(self.device)

    self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
    self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
    self._color_vol = torch.zeros(*self._vol_dim).to(self.device)


  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
    im_h, im_w = depth_im.shape

    cam_pose = torch.from_numpy(cam_pose).float().to(self.device)
    cam_intr = torch.from_numpy(cam_intr).float().to(self.device)
    color_im = torch.from_numpy(color_im).float().to(self.device)
    depth_im = torch.from_numpy(depth_im).float().to(self.device)

    return integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight,
                     self._world_c,
                     self._vox_coords,
                     self._weight_vol,
                     self._tsdf_vol,
                     self._color_vol,
                     self._sdf_trunc,
                     im_h, im_w)

  def finalize(self):
    self._tsdf_vol = self._tsdf_vol.cpu().numpy()
    self._color_vol = self._color_vol.cpu().numpy()


def integrate(
  color_im,
  depth_im,
  cam_intr,
  cam_pose,
  obs_weight: float,
  world_c,
  vox_coords,
  weight_vol,
  tsdf_vol,
  color_vol,
  sdf_trunc: float,
  im_h: int,
  im_w: int,
):
  const_val = 256*256

  # Fold RGB color image into a single channel image
  color_im = torch.floor(color_im[..., 2]*256*256 + color_im[..., 1]*256 + color_im[..., 0])

  # Convert world coordinates to camera coordinates
  world2cam = torch.inverse(cam_pose)
  cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

  # Convert camera coordinates to pixel coordinates
  fx, fy = cam_intr[0, 0], cam_intr[1, 1]
  cx, cy = cam_intr[0, 2], cam_intr[1, 2]
  pix_z = cam_c[:, 2]
  pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
  pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

  # Eliminate pixels outside view frustum
  valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
  valid_vox_x = vox_coords[valid_pix, 0]
  valid_vox_y = vox_coords[valid_pix, 1]
  valid_vox_z = vox_coords[valid_pix, 2]
  valid_pix_y = pix_y[valid_pix]
  valid_pix_x = pix_x[valid_pix]
  depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

  # Integrate tsdf
  depth_diff = depth_val - pix_z[valid_pix]
  dist = torch.clamp(depth_diff / sdf_trunc, max=1)
  valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
  valid_vox_x = valid_vox_x[valid_pts]
  valid_vox_y = valid_vox_y[valid_pts]
  valid_vox_z = valid_vox_z[valid_pts]
  valid_pix_y = valid_pix_y[valid_pts]
  valid_pix_x = valid_pix_x[valid_pts]
  valid_dist = dist[valid_pts]
  w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  w_new = w_old + obs_weight
  tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight*valid_dist) / w_new
  weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

  # Integrate color
  old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  old_b = torch.floor(old_color / const_val)
  old_g = torch.floor((old_color-old_b*const_val) / 256)
  old_r = old_color - old_b*const_val - old_g*256
  new_color = color_im[valid_pix_y, valid_pix_x]
  new_b = torch.floor(new_color / const_val)
  new_g = torch.floor((new_color - new_b*const_val) / 256)
  new_r = new_color - new_b*const_val - new_g*256
  new_b = torch.clamp(torch.round((w_old*old_b + obs_weight*new_b) / w_new), max=255)
  new_g = torch.clamp(torch.round((w_old*old_g + obs_weight*new_g) / w_new), max=255)
  new_r = torch.clamp(torch.round((w_old*old_r + obs_weight*new_r) / w_new), max=255)
  color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*const_val + new_g*256 + new_r

  return weight_vol, tsdf_vol, color_vol