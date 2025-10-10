NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo

SRC_DIR := src
BUILD_DIR := build
TARGETS := kepler_shfl occupancy_api

.PHONY: all
all: $(addprefix $(BUILD_DIR)/,$(TARGETS))

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean
clean: 
	@rm -rf $(BUILD_DIR)
