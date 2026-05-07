NVCC     := nvcc
CXX      := g++

# Override GPU arch at build time: make ARCH=sm_86
ARCH     ?= sm_75

NVCC_FLAGS := -O2 -std=c++17 -arch=$(ARCH) -Iinclude
CXX_FLAGS  := -O2 -std=c++17 -Iinclude

TARGET   := reduction
BUILD    := build

SRC_CU   := src/gpu_reduction.cu src/main.cu
SRC_CPP  := src/cpu_reduction.cpp src/data_gen.cpp

OBJ_CU   := $(patsubst src/%.cu,  $(BUILD)/%.cu.o,  $(SRC_CU))
OBJ_CPP  := $(patsubst src/%.cpp, $(BUILD)/%.cpp.o, $(SRC_CPP))
OBJS     := $(OBJ_CU) $(OBJ_CPP)

.PHONY: all clean run plot

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD)/%.cu.o: src/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

$(BUILD)/%.cpp.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

run: $(TARGET)
	./$(TARGET)

plot:
	python3 scripts/plot_results.py

clean:
	rm -rf $(BUILD) $(TARGET) results.csv
