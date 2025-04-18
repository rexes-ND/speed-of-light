all: build/gmem_access build/smem build/matrix_transpose

build/gmem_access: src/gmem_access.cu
	mkdir -p build
	nvcc src/gmem_access.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/gmem_access

build/smem: src/smem.cu
	mkdir -p build
	nvcc src/smem.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/smem

build/matrix_transpose: src/matrix_transpose.cu
	mkdir -p build
	nvcc src/matrix_transpose.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/matrix_transpose

clean: 
	rm -rf build
