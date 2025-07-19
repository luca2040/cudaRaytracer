# === CONFIG ===
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++17
NVCCFLAGS = -Xcompiler "-Wall -std=c++17"
LDFLAGS = -lSDL2 -lSDL2_ttf -lpthread

LDFLAGS += -lGLEW -lGL

# Optimizations
# CXXFLAGS += -fopenmp
# LDFLAGS  += -Xcompiler=-fopenmp -lgomp
CXXFLAGS += -O3 -march=native -ffast-math
NVCCFLAGS += -O3 --use_fast_math --fmad=true -ftz=true --relocatable-device-code=false
NVCCFLAGS += -Xptxas -O3,-warn-spills,-v
NVCCFLAGS += -arch=native
# NVCCFLAGS += -gencode arch=compute_86,code=sm_86

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/TheBest3dRendererEverRTX

# CPP_FILES += $(OPTIX_DIR)/SDK/optix/optix_stubs.cpp

# === Tracy Profiler ===
TRACY_DIR = $(SRC_DIR)/third_party/tracy
TRACY_CPP = $(TRACY_DIR)/TracyClient.cpp
CXXFLAGS += -DTRACY_ENABLE -DTRACY_NO_RPMALLOC -DTRACY_NO_CALLSTACK -I$(TRACY_DIR)
NVCCFLAGS += -DTRACY_ENABLE -DTRACY_NO_RPMALLOC -DTRACY_NO_CALLSTACK -I$(TRACY_DIR)

# === FILES ===
TRACY_CPP = $(SRC_DIR)/third_party/tracy/TracyClient.cpp

CPP_FILES := $(shell find $(SRC_DIR) -name '*.cpp' ! -path "$(SRC_DIR)/third_party/tracy/*")
CPP_FILES += $(TRACY_CPP)

CU_FILES  := $(shell find $(SRC_DIR) -name '*.cu')
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_FILES)) \
             $(patsubst $(SRC_DIR)/%.cu,  $(BUILD_DIR)/%.o, $(CU_FILES))

# === BUILD RULES ===

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(NVCC) $(OBJ_FILES) -o $@ $(LDFLAGS)

# Compile C++ files with g++
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Building $@ from $<"
	@mkdir -p $(dir $@)

	@if echo "$<" | grep -q "third_party"; then \
		$(CXX) -w $(filter-out -Wall,$(CXXFLAGS)) -c $< -o $@ ; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@ ; \
	fi

# Compile CUDA files with nvcc
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Building $@ from $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run
run: all
	./$(TARGET)

# Clean
clean:
	rm -rf $(BUILD_DIR)

clear: clean

.PHONY: runf crunf

# Run fast (multithreaded compilation)
runf:
	$(MAKE) -j$(shell nproc) run

# Clear and run fast
crunf:
	$(MAKE) clear && $(MAKE) runf