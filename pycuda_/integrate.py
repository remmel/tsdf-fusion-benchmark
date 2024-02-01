import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

from fusion import TSDFVolumeBase

def to_gpu(data):
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)
    return data_gpu

class TSDFVolumePycuda(TSDFVolumeBase):
    using= 'Using PyCUDA'
    useGPU = True

    def prepare(self):
        self._tsdf_vol_gpu = to_gpu(self._tsdf_vol)
        self._weight_vol_gpu = to_gpu(self._weight_vol)
        self._color_vol_gpu = to_gpu(self._color_vol)

        with open('pycuda_/integrate_kernel.cu', 'r') as file:
            kernel_code = file.read()
        # Cuda kernel function (C++)
        self._cuda_src_mod = SourceModule(kernel_code)

        self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        self._n_gpu_loops = int(np.ceil(
            float(np.prod(self._vol_dim)) / float(np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * self._const + color_im[..., 1] * 256 + color_im[..., 0])

        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(self._tsdf_vol_gpu,
                                 self._weight_vol_gpu,
                                 self._color_vol_gpu,
                                 cuda.InOut(self._vol_dim.astype(np.float32)),
                                 cuda.InOut(self._vol_origin.astype(np.float32)),
                                 cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                 cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                 cuda.InOut(np.asarray([
                                     gpu_loop_idx,
                                     self._voxel_size,
                                     im_h,
                                     im_w,
                                     self._sdf_trunc,
                                     obs_weight
                                 ], np.float32)),
                                 cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                 block=(self._max_gpu_threads_per_block, 1, 1),
                                 grid=(
                                     int(self._max_gpu_grid_dim[0]),
                                     int(self._max_gpu_grid_dim[1]),
                                     int(self._max_gpu_grid_dim[2]),
                                 )
                                 )

    def finalize(self):
        cuda.memcpy_dtoh(self._tsdf_vol, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol, self._color_vol_gpu)
