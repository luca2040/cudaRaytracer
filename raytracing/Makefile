# === CONFIG ===
CXX = g++
CXXFLAGS = -Wall -std=c++17
LDFLAGS = -lSDL2 -lSDL2_ttf

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/TheBest3dRendererEverRTX

SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))

# === BUILD RULES ===

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run
run: all
	./$(TARGET)

# Clean
clean:
	rm -rf $(BUILD_DIR)
clear:
	rm -rf $(BUILD_DIR)
