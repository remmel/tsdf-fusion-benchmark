from fusion import TSDFVolumeBase
import taichi as ti
import numpy as np

@ti.data_oriented
class TSDFVolumeTaichi(TSDFVolumeBase):
    using = "Using Taichi"

    def __init__(self, vol_bnds, voxel_size, useGPU=True):
        self.useGPU = useGPU
        # ti.gpu ti.cuda ti.opengl
        ti.init(arch=ti.opengl if self.useGPU else ti.cpu)

        super().__init__(vol_bnds, voxel_size)

    def prepare(self):
        # # Taichi fields to replace numpy arrays from the parent class
        self.tsdf_vol = ti.field(dtype=ti.f64, shape=self._tsdf_vol.shape)
        self.weight_vol = ti.field(dtype=ti.f64, shape=self._weight_vol.shape)
        self.color_vol = ti.field(dtype=ti.f64, shape=self._color_vol.shape)
        self.tsdf_vol.from_numpy(self._tsdf_vol)
        self.weight_vol.from_numpy(self._weight_vol)
        self.color_vol.from_numpy(self._color_vol)

    @ti.kernel
    def integrate_kernel(self, color_im: ti.types.ndarray(), depth_im: ti.types.ndarray(),
                         cam_intr: ti.types.ndarray(), cam_pose_inv: ti.types.ndarray(),
                         obs_weight: ti.f32, _vol_dim: ti.types.ndarray(), _vol_origin: ti.types.ndarray(),
                         _voxel_size: ti.f32, _trunc_margin: ti.f32):
        for x, y, z in ti.ndrange(self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]):
            # Voxel grid coordinates to world coordinates
            world_x = _vol_origin[0] + x * _voxel_size
            world_y = _vol_origin[1] + y * _voxel_size
            world_z = _vol_origin[2] + z * _voxel_size

            # World to camera coordinates
            cam_x = cam_pose_inv[0, 0] * world_x + cam_pose_inv[0, 1] * world_y + cam_pose_inv[0, 2] * world_z + cam_pose_inv[0, 3]
            cam_y = cam_pose_inv[1, 0] * world_x + cam_pose_inv[1, 1] * world_y + cam_pose_inv[1, 2] * world_z + cam_pose_inv[1, 3]
            cam_z = cam_pose_inv[2, 0] * world_x + cam_pose_inv[2, 1] * world_y + cam_pose_inv[2, 2] * world_z + cam_pose_inv[2, 3]

            if cam_z > 0:
                pixel_x = int(cam_intr[0, 0] * cam_x / cam_z + cam_intr[0, 2])
                pixel_y = int(cam_intr[1, 1] * cam_y / cam_z + cam_intr[1, 2])

                if 0 <= pixel_x < depth_im.shape[1] and 0 <= pixel_y < depth_im.shape[0]:
                    depth_value = depth_im[pixel_y, pixel_x]
                    if depth_value != 0:
                        depth_diff = depth_value - cam_z
                        if -_trunc_margin < depth_diff < _trunc_margin:
                            dist = min(1.0, depth_diff / _trunc_margin)
                            w_old = self.weight_vol[x, y, z]
                            w_new = w_old + obs_weight
                            self.tsdf_vol[x, y, z] = (self.tsdf_vol[x, y, z] * w_old + obs_weight * dist) / w_new
                            self.weight_vol[x, y, z] = w_new
                            # Update color volume
                            old_color = self.color_vol[x, y, z]
                            new_color = color_im[pixel_y, pixel_x]
                            self.color_vol[x, y, z] = new_color  # Simplified update

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
        # Convert inputs to the necessary format
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * 256 * 256 + color_im[..., 1] * 256 + color_im[..., 0])

        # Convert depth, intrinsics, and pose
        cam_pose_inv = np.linalg.inv(cam_pose)

        # Call the kernel
        self.integrate_kernel(color_im, depth_im, cam_intr, cam_pose_inv, obs_weight,
                              self._vol_dim, self._vol_origin, self._voxel_size, self._sdf_trunc)

    def finalize(self):
        self._tsdf_vol = self.tsdf_vol.to_numpy()
        self._weight_vol = self.weight_vol.to_numpy()
        self._color_vol = self.color_vol.to_numpy()
