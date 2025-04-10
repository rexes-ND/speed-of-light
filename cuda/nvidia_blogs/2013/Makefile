all: build/gmem_access

build/gmem_access: src/gmem_access.cu
	mkdir -p build
	nvcc src/gmem_access.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/gmem_access

clean: 
	rm -rf build
