from typing import Tuple

import numpy as np
from numba import cuda

from fusion import TSDFVolume


class TSDFVolumeChild(TSDFVolume):
    using = "Using numba cuda (@cuda.jit)"

    def prepare(self):
        self._weight_vol = cuda.to_device(self._weight_vol)
        self._tsdf_vol = cuda.to_device(self._tsdf_vol)
        self._color_vol = cuda.to_device(self._color_vol)

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
        im_h, im_w = depth_im.shape

        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * 256 * 256 + color_im[..., 1] * 256 + color_im[..., 0])
        color_im = cuda.to_device(color_im)
        depth_im = cuda.to_device(depth_im)
        cam_intr = cuda.to_device(cam_intr)
        cam_pose_inv = cuda.to_device(np.linalg.inv(cam_pose))

        threads_per_block = 64
        blocks_per_grid = (self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2] + (threads_per_block - 1)) // threads_per_block

        integrate_kernel[blocks_per_grid, threads_per_block](color_im, depth_im, cam_intr, cam_pose_inv, obs_weight,
                                                             self._vol_dim, self._vol_origin,
                                                             self._voxel_size, self._sdf_trunc,
                                                             self._weight_vol, self._tsdf_vol, self._color_vol)

    def finalize(self):
        self._weight_vol = self._weight_vol.copy_to_host()
        self._tsdf_vol = self._tsdf_vol.copy_to_host()
        self._color_vol = self._color_vol.copy_to_host()



@cuda.jit()
def integrate_kernel(color_im, depth_im, cam_intr, cam_pose_inv, obs_weight, _vol_dim, _vol_origin, _voxel_size,
                     _trunc_margin, _weight_vol, _tsdf_vol, _color_vol):
  # Calculate the unique thread index
  i = cuda.grid(1)

  vol_size = _vol_dim[0] * _vol_dim[1] * _vol_dim[2]
  if i < vol_size:
    # Get voxel grid coordinates
    z = i // (_vol_dim[0] * _vol_dim[1])
    y = (i % (_vol_dim[0] * _vol_dim[1])) // _vol_dim[0]
    x = i % _vol_dim[0]

    # Voxel grid coordinates to world coordinates
    world_x = _vol_origin[0] + x * _voxel_size
    world_y = _vol_origin[1] + y * _voxel_size
    world_z = _vol_origin[2] + z * _voxel_size

    # World coordinates to camera coordinates - cam_pose_inv @ world_c_xyz1
    cam_x = cam_pose_inv[0, 0] * world_x + cam_pose_inv[0, 1] * world_y + cam_pose_inv[0, 2] * world_z + cam_pose_inv[0, 3]
    cam_y = cam_pose_inv[1, 0] * world_x + cam_pose_inv[1, 1] * world_y + cam_pose_inv[1, 2] * world_z + cam_pose_inv[1, 3]
    cam_z = cam_pose_inv[2, 0] * world_x + cam_pose_inv[2, 1] * world_y + cam_pose_inv[2, 2] * world_z + cam_pose_inv[2, 3]

    # Project camera coordinates to image plane
    pixel_x = int(round(cam_intr[0, 0] * (cam_x / cam_z) + cam_intr[0, 2]))
    pixel_y = int(round(cam_intr[1, 1] * (cam_y / cam_z) + cam_intr[1, 2]))

    # Skip if outside view frustum
    if 0 <= pixel_x < depth_im.shape[1] and 0 <= pixel_y < depth_im.shape[0] and cam_z > 0:
      depth_value = depth_im[pixel_y, pixel_x]

      # Skip invalid depth
      if depth_value != 0:
        depth_diff = depth_value - cam_z
        if -_trunc_margin < depth_diff < _trunc_margin:
          dist = min(1.0, depth_diff / _trunc_margin)
          w_old = _weight_vol[x, y, z]
          w_new = w_old + obs_weight
          _tsdf_vol[x, y, z] = (_tsdf_vol[x, y, z] * w_old + obs_weight * dist) / w_new
          _weight_vol[x, y, z] = w_new

          # Update color volume
          old_color = _color_vol[x, y, z]
          new_color = color_im[pixel_y, pixel_x]
          _color_vol[x, y, z] = new_color  # Simplified color update