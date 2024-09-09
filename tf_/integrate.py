import tensorflow as tf
import numpy as np
from fusion import TSDFVolumeBase


class TSDFVolumeTf(TSDFVolumeBase):
    using = "Using TensorFlow"

    def __init__(self, vol_bnds, voxel_size, useGPU=True):
        self.useGPU = useGPU
        self.device = "/gpu:0" if self.useGPU else "/cpu:0"

        super().__init__(vol_bnds, voxel_size)

    def prepare(self):
        self.initVoxCoords()
        self.initWorldCoords()

        with tf.device(self.device):
            self._world_c = tf.constant(self._world_c, dtype=tf.float32)
            self._vox_coords = tf.constant(self._vox_coords, dtype=tf.int32)

            self._weight_vol = tf.zeros(self._vol_dim, dtype=tf.float32)
            self._tsdf_vol = tf.ones(self._vol_dim, dtype=tf.float32)
            self._color_vol = tf.zeros(self._vol_dim, dtype=tf.float32)

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
        im_h, im_w = depth_im.shape

        with tf.device(self.device):
            cam_pose = tf.constant(cam_pose, dtype=tf.float32)
            cam_intr = tf.constant(cam_intr, dtype=tf.float32)
            color_im = tf.constant(color_im, dtype=tf.float32)
            depth_im = tf.constant(depth_im, dtype=tf.float32)

            self._weight_vol, self._tsdf_vol, self._color_vol = self.integrate_static(
                color_im,
                depth_im,
                cam_intr,
                cam_pose,
                obs_weight,
                self._world_c,
                self._vox_coords,
                self._weight_vol,
                self._tsdf_vol,
                self._color_vol,
                self._sdf_trunc,
                im_h,
                im_w,
            )

    def finalize(self):
        self._tsdf_vol = self._tsdf_vol.numpy()
        self._color_vol = self._color_vol.numpy()

    @staticmethod
    @tf.function
    def integrate_static(
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
        const_val = 256 * 256

        # Fold RGB color image into a single channel image
        color_im = tf.floor(color_im[..., 2] * 256 * 256 + color_im[..., 1] * 256 + color_im[..., 0])

        # Convert world coordinates to camera coordinates
        world2cam = tf.linalg.inv(cam_pose)
        cam_c = tf.transpose(tf.matmul(world2cam, tf.transpose(world_c)))

        # Convert camera coordinates to pixel coordinates
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]
        pix_z = cam_c[:, 2]
        pix_x = tf.cast(tf.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx), dtype=tf.int32)
        pix_y = tf.cast(tf.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy), dtype=tf.int32)

        # Eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_x = vox_coords[valid_pix][:, 0]
        valid_vox_y = vox_coords[valid_pix][:, 1]
        valid_vox_z = vox_coords[valid_pix][:, 2]
        valid_pix_y = pix_y[valid_pix]
        valid_pix_x = pix_x[valid_pix]
        depth_val = tf.gather_nd(depth_im, tf.stack([valid_pix_y, valid_pix_x], axis=1))

        # Integrate tsdf
        depth_diff = depth_val - tf.boolean_mask(pix_z, valid_pix)
        dist = tf.clip_by_value(depth_diff / sdf_trunc, -1, 1)
        valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
        valid_vox_x = tf.boolean_mask(valid_vox_x, valid_pts)
        valid_vox_y = tf.boolean_mask(valid_vox_y, valid_pts)
        valid_vox_z = tf.boolean_mask(valid_vox_z, valid_pts)
        valid_pix_y = tf.boolean_mask(valid_pix_y, valid_pts)
        valid_pix_x = tf.boolean_mask(valid_pix_x, valid_pts)
        valid_dist = tf.boolean_mask(dist, valid_pts)
        w_old = tf.gather_nd(weight_vol, tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1))
        tsdf_vals = tf.gather_nd(tsdf_vol, tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1))
        w_new = w_old + obs_weight
        tsdf_vol = tf.tensor_scatter_nd_update(
            tsdf_vol,
            tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1),
            (w_old * tsdf_vals + obs_weight * valid_dist) / w_new,
        )
        weight_vol = tf.tensor_scatter_nd_update(
            weight_vol, tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1), w_new
        )

        # Integrate color
        old_color = tf.gather_nd(color_vol, tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1))
        old_b = tf.floor(old_color / const_val)
        old_g = tf.floor((old_color - old_b * const_val) / 256)
        old_r = old_color - old_b * const_val - old_g * 256
        new_color = tf.gather_nd(color_im, tf.stack([valid_pix_y, valid_pix_x], axis=-1))
        new_b = tf.floor(new_color / const_val)
        new_g = tf.floor((new_color - new_b * const_val) / 256)
        new_r = new_color - new_b * const_val - new_g * 256
        new_b = tf.clip_by_value(tf.round((w_old * old_b + obs_weight * new_b) / w_new), 0, 255)
        new_g = tf.clip_by_value(tf.round((w_old * old_g + obs_weight * new_g) / w_new), 0, 255)
        new_r = tf.clip_by_value(tf.round((w_old * old_r + obs_weight * new_r) / w_new), 0, 255)
        color_vol = tf.tensor_scatter_nd_update(
            color_vol,
            tf.stack([valid_vox_x, valid_vox_y, valid_vox_z], axis=-1),
            new_b * const_val + new_g * 256 + new_r,
        )

        return weight_vol, tsdf_vol, color_vol