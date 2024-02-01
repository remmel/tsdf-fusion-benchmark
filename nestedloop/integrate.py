from typing import Tuple

import numpy as np
from numba import njit, prange

from fusion import TSDFVolumeBase


class TSDFVolumeNestedLoop(TSDFVolumeBase):
  using = "Using nested loop"
  useGPU = False #Handle only CPU

  def prepare(self):
    self.initVoxCoords()
    self.initWorldCoords()

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
    im_h, im_w = depth_im.shape

    self.integrate_static(color_im, depth_im, np.ascontiguousarray(cam_intr), np.ascontiguousarray(cam_pose), obs_weight,
              self._world_c,
              self._vox_coords,
              self._weight_vol, self._tsdf_vol,  self._color_vol,  # are modified
              self._sdf_trunc,
              im_h, im_w,
              self._vol_dim,
              self._vol_origin,
              self._voxel_size)

  def finalize(self):
    pass

  #@njit(parallel=True) prange for z (min: 0.097)

  @staticmethod
  @njit(parallel=True)
  def integrate_static(color_im, depth_im, cam_intr, cam_pose, obs_weight: float,
        world_c,
        vox_coords,
        weight_vol, tsdf_vol, color_vol,
        sdf_trunc: float,
        im_h: int, im_w: int,
        vol_dim: Tuple[int, int, int],
        vol_origin: Tuple[float, float, float],
        voxel_size: float,
  ):
    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[..., 2] * 256 * 256 + color_im[..., 1] * 256 + color_im[..., 0])

    vol_dim_x, vol_dim_y, vol_dim_z = vol_dim

    cam_pose_inv = np.linalg.inv(cam_pose) # world2cam

    for x in prange(vol_dim_x):
      for y in range(vol_dim_y):
        for z in range(vol_dim_z):
          # Voxel grid coordinates to world coordinates (or done in init, 44% quicker)
          # world_c_xyz = vol_origin + np.array([x,y,z]) * voxel_size
          # world_c_xyz1 = np.append(world_c_xyz, 1.0)

          idx = z + y * vol_dim_z + x * vol_dim_z * vol_dim_y
          world_c_xyz1 = world_c[idx]

          # World coordinates to camera coordinates
          cam_c = cam_pose_inv @ world_c_xyz1
          cam_x, cam_y, cam_z = cam_c[0], cam_c[1], cam_c[2]

          # Project camera coordinates to image plane
          fx, fy = cam_intr[0, 0], cam_intr[1, 1]
          cx, cy = cam_intr[0, 2], cam_intr[1, 2]
          pixel_x = int(np.round(fx * (cam_x / cam_z) + cx))
          pixel_y = int(np.round(fy * (cam_y / cam_z) + cy))

          # Skip if outside view frustum
          if 0 <= pixel_x < depth_im.shape[1] and 0 <= pixel_y < depth_im.shape[0] and cam_z > 0:
            depth_value = depth_im[pixel_y, pixel_x]

            # Skip invalid depth
            if depth_value != 0:
              depth_diff = depth_value - cam_z
              if -sdf_trunc < depth_diff < sdf_trunc:
                dist = min(1.0, depth_diff / sdf_trunc)
                w_old = weight_vol[x, y, z]
                w_new = w_old + obs_weight
                tsdf_vol[x, y, z] = (tsdf_vol[x, y, z] * w_old + obs_weight * dist) / w_new
                weight_vol[x, y, z] = w_new

                # Update color volume
                old_color = color_vol[x, y, z]
                new_color = color_im[pixel_y, pixel_x]
                color_vol[x, y, z] = new_color  # Simplified color update

    return weight_vol, tsdf_vol, color_vol