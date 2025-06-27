# === CONFIG ===
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++17
NVCCFLAGS = -Xcompiler "-Wall -std=c++17"
LDFLAGS = -lSDL2 -lSDL2_ttf

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/TheBest3dRendererEverRTX

# === FILES ===
CPP_FILES := $(shell find $(SRC_DIR) -name '*.cpp')
CU_FILES  := $(shell find $(SRC_DIR) -name '*.cu')
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_FILES)) \
             $(patsubst $(SRC_DIR)/%.cu,  $(BUILD_DIR)/%.o, $(CU_FILES))

# === BUILD RULES ===

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(NVCC) $(OBJ_FILES) -o $@ $(LDFLAGS)

# Compile C++ files with g++
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files with nvcc
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run
run: all
	./$(TARGET)

# Clean
clean:
	rm -rf $(BUILD_DIR)

clear: clean
