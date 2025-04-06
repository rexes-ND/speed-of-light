all: build/easy_intro_to_cuda build/performance_metrics build/query_dev_handle_error build/data_transfer

build/easy_intro_to_cuda: src/easy_intro_to_cuda.cu
	mkdir -p build
	nvcc src/easy_intro_to_cuda.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/easy_intro_to_cuda

build/performance_metrics: src/performance_metrics.cu
	mkdir -p build
	nvcc src/performance_metrics.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/performance_metrics

build/query_dev_handle_error: src/query_dev_handle_error.cu
	mkdir -p build
	nvcc src/query_dev_handle_error.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/query_dev_handle_error

build/data_transfer: src/data_transfer.cu
	mkdir -p build
	nvcc src/data_transfer.cu -std=c++17 -gencode arch=compute_89,code=sm_89 -o build/data_transfer

clean:
	rm -rf build
