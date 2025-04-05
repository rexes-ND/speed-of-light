all: build/easy_intro_to_cuda build/performance_metrics

build/easy_intro_to_cuda: src/easy_intro_to_cuda.cu
	mkdir -p build
	nvcc src/easy_intro_to_cuda.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/easy_intro_to_cuda

build/performance_metrics: src/performance_metrics.cu
	mkdir -p build
	nvcc src/performance_metrics.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/performance_metrics

clean:
	rm -rf build
