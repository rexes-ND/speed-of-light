all: build/gmem_access build/smem

build/gmem_access: src/gmem_access.cu
	mkdir -p build
	nvcc src/gmem_access.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/gmem_access

build/smem: src/smem.cu
	mkdir -p build
	nvcc src/smem.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/smem

clean: 
	rm -rf build
