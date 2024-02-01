import numpy as np

from skimage import measure

import torch


class TSDFVolumeBase:
  """Volumetric TSDF Fusion of RGB-D Images.
  """

  using = None #Name of the implementatio
  useGpu = None #Whether that impl uses GPU or not

  def __init__(self, vol_bnds, voxel_size:float, useGPU:bool = None):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
      useGpu (bool): Some implementations handle both CPU and GPU system.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._sdf_trunc = 5 * self._voxel_size
    self._const = 256*256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
    self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)
    self._num_voxels = np.prod(self._vol_dim)

    print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
    print("[*] num voxels: {:,}".format(self._num_voxels))

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol = np.zeros(self._vol_dim).astype(np.float32)

    self.prepare()

  def initVoxCoords(self):
    # Get voxel grid coordinates
    xv, yv, zv = np.meshgrid(
      range(self._vol_dim[0]),
      range(self._vol_dim[1]),
      range(self._vol_dim[2]),
      indexing='ij'
    )
    self._vox_coords = np.concatenate([
      xv.reshape(1, -1),
      yv.reshape(1, -1),
      zv.reshape(1, -1)
    ], axis=0).astype(int).T

  def initWorldCoords(self):
    # Convert voxel coordinates to world coordinates
    # In Andy repo, this is done in integrate
    self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
    self._world_c = np.concatenate([self._world_c, np.ones((len(self._world_c), 1))], axis=1)


  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
    """Integrate an RGB-D frame into the TSDF volume.
    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign to the current observation.
    """
    raise Exception('must be implemented')


  def extract_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """

    tsdf_vol = self._tsdf_vol
    color_vol =  self._color_vol
    vol_origin = self._vol_origin

    # Marching cubes#
    verts = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._const)
    colors_g = np.floor((rgb_vals - colors_b*self._const) / 256)
    colors_r = rgb_vals - colors_b*self._const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def extract_triangle_mesh(self):
    """Extract a triangle mesh from the voxel volume using marching cubes.
    """
    tsdf_vol = self._tsdf_vol
    color_vol = self._color_vol
    vol_origin = self._vol_origin

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._const)
    colors_g = np.floor((rgb_vals - colors_b*self._const) / 256)
    colors_r = rgb_vals - colors_b*self._const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    return verts, faces, norms, colors

  @property
  def sdf_trunc(self):
    return self._sdf_trunc

  @property
  def voxel_size(self):
    return self._voxel_size


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = (transform @ xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
