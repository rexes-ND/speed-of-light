## [CUDA Pro Tip: Flush Denormals with Confidence](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/)

This blog is about denormal floating point numbers.
The denormals allows what is known as "gradual underflow" when a result is too small, and helps avoid catastrophic division-by zero errors.
Gradual or graceful underflow is using denormals to let number fade smoothly toward 0 instead of abruptly snapping to zero.
In GPU, there are some cases where it has to take a slower path for denormal values.
One way to avoid denormals is to add small "noise" so that the number is guaranteed to be denormal.
`nvcc` provides the cmdline option `-ftz=true` which causes all denormalized numbers to be flushed to zero.

- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](src/gmem_access.cu)
- [Using Shared Memory in CUDA C/C++](src/smem.cu)
- [An Efficient Matrix Transpose in CUDA C/C++](src/matrix_transpose.cu)
