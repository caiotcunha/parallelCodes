# Compiladores
# Tenta usar o g++ do ambiente ou do sistema
CC := /usr/bin/g++-11
NVCC := nvcc

ifndef CONDA_PREFIX
    $(error CONDA_PREFIX não está definido. Ative seu ambiente conda (conda activate nome_env))
endif

ARCH_FLAGS := -arch=compute_75 -code=sm_75

# Flags de Compilação
CFLAGS := -O3 $(ARCH_FLAGS) -std=c++14 --compiler-options -Wall

# Diretórios
INCLUDE_DIRS := -I./include \
                -I$(CONDA_PREFIX)/include/opencv4 \
                -I$(CONDA_PREFIX)/include

LIB_DIRS := -L$(CONDA_PREFIX)/lib

LIBS := -lnppig -lnppidei -lnpps -lnppc -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lcudart

BUILD_DIR := ./build
SRC_DIR := ./src
TARGET := exec

SRCS := $(SRC_DIR)/main.cu
OBJS := $(BUILD_DIR)/main.o


all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH_FLAGS) -ccbin $(CC) -o $@ $^ $(LIB_DIRS) $(LIBS)

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) -x cu -c $(CFLAGS) $(INCLUDE_DIRS) -ccbin $(CC) -o $@ $<

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)/*.o
	# Se quiser remover a pasta build inteira: rm -rf $(BUILD_DIR)