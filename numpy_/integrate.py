import numpy as np
from fusion import TSDFVolume


class TSDFVolumeChild(TSDFVolume):
    using = "Using numpy vectorized array"

    def prepare(self):
        self.initVoxCoords()
        self.initWorldCoords()

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
        im_h, im_w = depth_im.shape

        integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight,
                  self._world_c,
                  self._vox_coords,
                  self._weight_vol,
                  self._tsdf_vol,  # is modified
                  self._color_vol, # is modified
                  self._sdf_trunc, # is modified
                  im_h, im_w)

    def finalize(self):
        pass

# @jit(parallel=True, fastmath= True)
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
    const_val = 256 * 256
    # Fold RGB color image into a single channel image
    color_im = color_im[..., 2] * np.uint64(256*256) + color_im[..., 1] * 256 + color_im[..., 0]

    # Convert world coordinates to camera coordinates
    world2cam = np.linalg.inv(cam_pose)
    cam_c = (world2cam @ world_c.T).T

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = np.round((cam_c[:, 0] * fx / pix_z) + cx).astype(np.int64)
    pix_y = np.round((cam_c[:, 1] * fy / pix_z) + cy).astype(np.int64)

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    valid_vox_x, valid_vox_y, valid_vox_z = vox_coords[valid_pix, 0], vox_coords[valid_pix, 1], vox_coords[valid_pix, 2]
    valid_pix_x, valid_pix_y = pix_x[valid_pix], pix_y[valid_pix]
    depth_val = depth_im[valid_pix_y, valid_pix_x]

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = np.clip(depth_diff / sdf_trunc, None, 1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    valid_vox_x, valid_vox_y, valid_vox_z = valid_vox_x[valid_pts], valid_vox_y[valid_pts], valid_vox_z[valid_pts]
    valid_pix_x, valid_pix_y = valid_pix_x[valid_pts], valid_pix_y[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    # Integrate color
    old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    old_b = np.floor(old_color / const_val)
    old_g = np.floor((old_color - old_b * const_val) / 256)
    old_r = old_color - old_b * const_val - old_g * 256
    new_color = color_im[valid_pix_y, valid_pix_x]
    new_b = np.floor(new_color / const_val)
    new_g = np.floor((new_color - new_b * const_val) / 256)
    new_r = new_color - new_b * const_val - new_g * 256
    new_b = np.clip(np.round((w_old * old_b + obs_weight * new_b) / w_new), 0, 255)
    new_g = np.clip(np.round((w_old * old_g + obs_weight * new_g) / w_new), 0, 255)
    new_r = np.clip(np.round((w_old * old_r + obs_weight * new_r) / w_new), 0, 255)
    color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * const_val + new_g * 256 + new_r

    return weight_vol, tsdf_vol, color_vol
