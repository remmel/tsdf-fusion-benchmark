import argparse
import time

import cv2
import numpy as np

import fusion

import platform
if platform.system() == "Darwin":
  print("Using macOS")
  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True'

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}


EXAMPLE_CHOICES = ['pycuda', 'pytorch', 'pytorchcpp', 'pytorchjit', 'cuda', 'numpy', 'nestedloop', 'numbacuda', 'taichi', 'tf']
def main(args):
  if args.example == 'pytorchcpp':
    from pytorchcpp.integrate import TSDFVolumePyTorchCpp as TSDFVolumeImpl
  elif args.example == 'pytorchjit':
    from pytorchjit.integrate import TSDFVolumePytorchJIT as TSDFVolumeImpl
  elif args.example == 'pytorch':
    from pytorch_.integrate import TSDFVolumePytorch as TSDFVolumeImpl
  elif args.example == 'numpy':
    from numpy_.integrate import TSDFVolumeNumpy as TSDFVolumeImpl
  elif args.example == 'nestedloop':
    from nestedloop.integrate import TSDFVolumeNestedLoop as TSDFVolumeImpl
  elif args.example == 'numbacuda':
    from numbacuda.integrate import TSDFVolumeNumbaCuda as TSDFVolumeImpl
  elif args.example == 'pycuda':
    from pycuda_.integrate import TSDFVolumePycuda as TSDFVolumeImpl
  elif args.example == 'taichi':
    from taichi_.integrate import TSDFVolumeTaichi as TSDFVolumeImpl
  elif args.example == 'tf':
    from tf_.integrate import TSDFVolumeTf as TSDFVolumeImpl
  else:
    raise Exception('unknown example: %s' % args.example)


  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 15
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = TSDFVolumeImpl(vol_bnds, 0.02, args.gpu)
  print(tsdf_vol.using, '- useGPU:', tsdf_vol.useGPU)

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  times = []
  for i in range(n_imgs):
    print("Fusing frame {:02d}/{:d}".format(i+1, n_imgs), end="")

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

    # Integrate observation into voxel volume (assume color aligned with depth)
    tic = time.time()
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    toc = time.time()
    times.append(toc-tic)
    print(" - {:.3f} s".format(toc-tic))

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  times = [t*TIME_SCALES[args.scale] for t in times]
  print("Integration time: avg: {:.3f} {} - min: {:.3f} - avg(w/o 1st): {:.3f}".format(np.mean(times), args.scale, np.min(times), np.mean(times[1:])))

  tsdf_vol.finalize()


  # Extract pointcloud
  point_cloud = tsdf_vol.extract_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.extract_triangle_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('example', choices=EXAMPLE_CHOICES)
  parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='s')
  parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=True)
  args = parser.parse_args()
  main(args)