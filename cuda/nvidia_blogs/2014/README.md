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

## [Adaptive Parallel Computation with CUDA Dynamic Parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)

[Code](https://github.com/canonizer/mandelbrot-dyn)

Dynamic Parallelism is advantaguous for parallel recursive algorithms or nested parallelism.

The blog introduces Dynamic Parellism by example using a fast hierarchical algorithm for computing images of the Mandlebrot set.

A straightforward way (per pixel computation) spends most of the computation time on pixels belonging to the set.

In general, the only areas where we need high-resolution computation are along the fractal boundary of the set.
This is what Mariani-Silver algorithm does.

```python
def mariani_silver(rectangle):
    if border(rectangle) has common dwell:
        fill recantangle with common dwell
    else if rectangle size < threshold:
        per-pixel computation
    else:
        for sub_rectangle in rectangle:
            mariani_silver(sub_rectangle)
```

Dynamic Parallelism uses the CUDA Device Runtime library (`cudadevrt`), a subset of CUDA Runtime API callable from device code.

To use Dynamic Parallelism, you must use a two-step separate compilation and linking process: first, compile your source into an object file, and then link the object file against the CUDA Device Runtime.

```bash
# compile
$ nvcc -arch=sm_89 -dc myprog.cu -o myprog.o

# link
$ nvcc -arch=sm_89 myprog.o -lcudadevrt -o myprog
```

## [CUDA Dynamic Parallelism API and Principles](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)

TODO

## [CUDA Pro Tip: Minimize the Tail Effect](https://developer.nvidia.com/blog/cuda-pro-tip-minimize-the-tail-effect/)

The Theoretical Occupancy is the number of threads which **may** run on each multiprocessor (SM).
The Achieved Occupancy is measured from the execution of the kernel.

On a NVIDIA Tesla K20, there are 13 SM(s) and the Theoretical Occupancy of my kernel was 4 blocks of 256 threads per SM (50%).

The total number of blocks that can run concurrently on a given GPU is referred to as Wave.
In our example, each full wave consists of 13 x 4 = 52 blocks.

Assuming that the kernel launched 128 blocks, there are 2 full waves and a much smaller wave of 24 blocks.
The last wave under-utilized the GPU but represent a significant fraction of the run time.

To improve the performance, the `__launch_bounds__` attribute is used to constrain the number of registers which was the main occupancy limiter.

As a result, the occupancy became 5 blocks of 256 threads per SM.
Each full wave consists of 13 x 5 = 65 blocks.
The same computation is achieved in 1 full wave and an almost-full wave of 63 blocks.

Tail effect is significant when the number of blocks executed for a kernel is small.
This is one of the reasons why launching a large number of blocks per grid is recommended.
