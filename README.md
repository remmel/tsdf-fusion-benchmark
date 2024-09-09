# torchsdf-fusion

My attempt at trying to outperform Andy Zeng's PyCuda [TSDF fusion](https://github.com/andyzeng/tsdf-fusion-python) implementation with PyTorch.

To build the PyTorch C++ integration function on macOS and Ubuntu use:

```
cd cpptorch
CXX=clang python setup.py install
```

Download dataset from https://github.com/andyzeng/tsdf-fusion-python

To run the benchmark, use:

```
# cuda kernel code with pyCuda
python benchmark.py pycuda

#pytorch (gpu & cpu)
python benchmark.py pytorch
python benchmark.py pytorch --no-gpu

# pytorch with jit @torch.jit.script (gpu & cpu)
python benchmark.py pytorchjit
python benchmark.py pytorchjit --no-gpu

# pytorchcpp (gpu & cpu)
python benchmark.py pytorchcpp
python benchmark.py pytorchcpp --no-gpu

# TODO, write cudakernel
# python benchmark.py cpptorchcudakernel

# numpy
python benchmark.py numpy

# nestedloop with numba (prange)
python benchmark.py nestedloop

python benchmark.py numbacuda
```

## Results


| Method           | FPS | Min Integration Time | GPU |
|------------------|-----|----------------------|-----|
| PyCuda-AndyZeng  | 98  | 0.006                | ✔  |
| PyTorch          | 35  | 0.014                | ✔  |
| PyTorch          | 3   | 0.290                | ❌  |
| PyTorch JIT      | 8   | 0.010                | ✔  |
| PyTorch JIT      | 3   | 0.298                | ❌  |
| PyTorch C++      | 33  | 0.015                | ✔  |
| PyTorch C++      | 3   | 0.322                | ❌  |
| Numpy vect       | 1   | 0.767                | ❌  |
| nestedloop/Numba | 2   | 0.183                | ❌  |
| Numba Cuda       | 20  | 0.009                | ✔  |
| Taichi           | 106 | 0.002                | ✔  |
| Taichi           | 32  | 0.021                | ❌  |



On some implementation, the 1st integration (or 4 1st integration for JIT Pytorch) takes more times, because the JIT need to compile; that's why the `min` is indicated.
FPS on only tested on 15 frames; should be done on more. It includes first run, which is slower (JIT compilation).
Tested on Laptop with AMD Ryzen 5 5600H and RTX3060 on Ubuntu 22.04

## Install

```shell
conda create -n tsdf-benchmark  python=3.10
conda activate tsdf-benchmark
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::pycuda

# conda env export | grep -v "^prefix: " > environment.yml #export to yml
# conda env create -f environment.yml #import yml
```

# TODO 
- Remove/rename the integrate file
- Add new implementation
  - Cupy?
  - mix of prange and vectorized numpy array
  - nestedloop w/o numba (really slow)
  - cpp nested loop
  - cpp torch cuda kernel (.cu)
- Remove implementation selection eg:
```python
  args.impl = 'PyTorch'
  module = importlib.import_module('impls')
  TSDFVolumeImpl = getattr(module, 'TSDFVolume' + args.impl)
```