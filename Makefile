# === ENABLE PROFILING VARIABLE ===
PROFILE ?= 0

# === CONFIG ===
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++17
NVCCFLAGS = -Xcompiler "-Wall -std=c++17"
LDFLAGS = -lSDL2 -lSDL2_ttf -lpthread
LDFLAGS += -lGLEW -lGL

# === OPTIMIZATIONS ===
CXXFLAGS += -O3 -march=native -ffast-math
NVCCFLAGS += -O3 --use_fast_math --fmad=true -ftz=true --relocatable-device-code=false
NVCCFLAGS += -Xptxas -O3,-warn-spills,-v
NVCCFLAGS += -arch=native

# === FILES ===
SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/TheBest3dRendererEverRTX

# === PROFILER ===
TRACY_DIR = $(SRC_DIR)/third_party/tracy
TRACY_CPP = $(TRACY_DIR)/TracyClient.cpp

ifeq ($(PROFILE),1)
	CXXFLAGS += -DTRACY_ENABLE -DTRACY_NO_RPMALLOC -DTRACY_NO_CALLSTACK -DENABLE_PROFILING -I$(TRACY_DIR)
	NVCCFLAGS += -DTRACY_ENABLE -DTRACY_NO_RPMALLOC -DTRACY_NO_CALLSTACK -DENABLE_PROFILING -I$(TRACY_DIR)
endif

# === FILES AGAIN ===
TRACY_CPP = $(SRC_DIR)/third_party/tracy/TracyClient.cpp

CPP_FILES := $(shell find $(SRC_DIR) -name '*.cpp' ! -path "$(SRC_DIR)/third_party/tracy/*")

ifeq ($(PROFILE),1)
	CPP_FILES += $(TRACY_CPP)
endif

CU_FILES  := $(shell find $(SRC_DIR) -name '*.cu')
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_FILES)) \
             $(patsubst $(SRC_DIR)/%.cu,  $(BUILD_DIR)/%.o, $(CU_FILES))

# === BUILD RULES ===
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(NVCC) $(OBJ_FILES) -o $@ $(LDFLAGS)

# === COMPILE C++ ===
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Building $@ from $<"
	@mkdir -p $(dir $@)

	@if echo "$<" | grep -q "third_party"; then \
		$(CXX) -w $(filter-out -Wall,$(CXXFLAGS)) -c $< -o $@ ; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@ ; \
	fi

# === COMPILE CUDA ===
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Building $@ from $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run
run: all
ifeq ($(PROFILE),1)
	./$(TARGET)
else
	@echo "\nAdd PROFILE=1 var to compile with profiler enabled\n"
	./$(TARGET)
endif

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