## [CUDA Pro Tip: Control GPU Visibility with CUDA_VISIBLE_DEVICES](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)

Use `CUDA_VISIBLE_DEVICES` for restricting execution to a specific device or set of devices.

TODO: Need some background for peer-to-peer memory access to fully understand this blog.

## [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)

[Code](src/kepler_shfl.cu)

The blog is about using warp shuffle functions to implement parallel reduction.

Atomic operations are also used and [CUB](https://nvidia.github.io/cccl/cub/) is mentioned.

## [Separate Compilation and Linking of CUDA C++ Device Code](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/)

[Code](src/separate_compilation_example/)

The blog explores separate compilation and linking of device code.

The `-dc` option tells `nvcc` to generate device code for later linking.
We omit `-dc` in the link command to tell `nvcc` to link the objects.
When `nvcc` is passed the object files with both CPU and GPU object code, it will link both automatically.

```bash
# Advanced Usage: Using g++ for the final link step

# Since g++ don't know how to link CUDA device code,
# nvcc has to link the CUDA device code using `-dlink`.
# This links all the device code and place it into gpuCode.o.
# Note that this doesn't link the CPU object code.
# In fact, the CPU object code in v3.o, particle.o, and main.o is discarded in this step.
$ nvcc -arch=sm_89 -dlink v3.o particle.o main.o -o gpuCode.o

# g++ need all of the objects again.
# The CUDA Runtime API library is automatically linked when we use nvcc for linking,
# but we must explicitly link it when using another linker.
$ g++ gpuCode.o main.o particle.o v3.o -L/usr/local/cuda/lib64 -lcudart -o app
```
