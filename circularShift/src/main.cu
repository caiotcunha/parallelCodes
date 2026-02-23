#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <omp.h>

#define TILE_DIM 64
#define BLOCK_ROWS 8


#define DEVICE_MALLOC_LIMIT 1024*1024*1024

// filename deve ser de um arquivo pgm
// Retorna um ponteiro para GPU
unsigned char* loadPGMToDevice(char* filename, int& width, int &height, size_t &pitch) {
  if(strcmp(&(filename[strlen(filename)-3]), "pgm")) {
    printf("Imagem inválida\n");
    exit(1);
  }

  // Variáveis auxiliares
  int i;
  unsigned char* d_imagem_row;

  // Abrimos a imagem
  FILE* imagem;
  imagem = fopen(filename, "rb");
  if(imagem==NULL) {
    printf("Falha ao ler a imagem\n");
    exit(1);
  }

  // Buffer de leitura do header
  char header_buffer[1024];

  // Variáveis do header
  char file_type[50];
  unsigned short max_val;

  // Lendo o header
  memset(header_buffer, 0, 1024);
  fgets(header_buffer, 1024, imagem);
  sscanf(header_buffer, "%s\n", file_type);

  // Lemos as dimensões da imagem
  memset(header_buffer, 0, strlen(header_buffer));
  fgets(header_buffer, 1024, imagem);
  if(sscanf(header_buffer, "%d %d\n", &height, &width) < 2) {
    printf("Falha ao ler dimsnesões da imagem\n");
    exit(1);
  }

  // Lemos o maior valor na imagem 
  memset(header_buffer, 0, strlen(header_buffer));
  fgets(header_buffer, 1024, imagem);
  sscanf(header_buffer, "%hu\n", &max_val); 

  if(max_val != 255) {
    printf("Pixels de 16-bits não são suportados\n");
    exit(1);
  }

  unsigned char* d_imagem;

  cudaError_t err = cudaMallocPitch(&d_imagem, &pitch, width*sizeof(uchar), height);
  if(err) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  // here maybe?
  uchar* buffer = (unsigned char*) malloc(width*sizeof(uchar)+32);
  uchar* rotatedBuffer = (unsigned char*) malloc(width*sizeof(uchar)+32); 
  for(i=0; i<height; i++) {
    if(fread(buffer, sizeof(unsigned char), width, imagem) < width) {
      printf("Falha na leitura (iteracao=%d)\n", i);
      exit(1);
    }

    // Rotating buffer
    int shift = i;
    int cols = width;
    for (int j = 0; j < width; ++j) {
        int coluna_origem = (j - shift + cols) % cols;
        rotatedBuffer[j] = buffer[coluna_origem];
    }
    
    d_imagem_row = (uchar*)((char*)d_imagem + i*pitch);
    err = cudaMemcpy(d_imagem_row, rotatedBuffer, width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    //err = cudaMemcpy(d_imagem_row, buffer, width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    if(err) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(1);
    }

  }
  

  fclose(imagem);
  free(buffer);

  return d_imagem;
}

void configureHeapSize() {
    // Aumentamos o heap para acomodar a fila
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_LIMIT);
}

int main(int argc, char** argv) {
    
    if(argc < 5) {
      printf("Numero insuficiente de argumentos (%d/4)\n", argc-1);
      return 1;
    }

    char* marker_filename = argv[1];
    char* mask_filename = argv[2];
    const char* output_filename = argv[3];
    int usePGMLoader = atoi(argv[4]);

    // Informações das imagens
    int size;
    int height;

    // Opencv container de imagem
    cv::Mat marker_img, mask_img;
    
    // Containers das imagens
    unsigned char* d_marker;
    size_t pitchMarker;
    unsigned char* d_mask;
    size_t pitchMask;
    
    // Configurando o heap do device
    configureHeapSize();
    
    // Lendo as imagens
	  // Usamos o carregamento customizado de pgm
    auto tStart = std::chrono::high_resolution_clock::now();
    if(usePGMLoader) {
      printf("Usando carregamento customizado de PGM\n");
      d_marker = loadPGMToDevice(marker_filename, size, height, pitchMarker);
    }
    // Usamos o opencv
    else {
      marker_img = cv::imread(marker_filename, cv::IMREAD_GRAYSCALE);
      
      if(marker_img.empty()) {
          printf("Falha ao ler o marcador\n");
          exit(1);
      }
      
      size = marker_img.size().width;
      // Alocamos a imagem
      cudaMallocPitch(&d_marker, &pitchMarker, size*sizeof(unsigned char), size);

      // Enviamos a imagem para a GPU
      cudaMemcpy2D(d_marker, pitchMarker, marker_img.data, marker_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);

    }

	  // Usamos o carregamento customizado de pgm
    if(usePGMLoader) {
      d_mask = loadPGMToDevice(mask_filename, size, height, pitchMask);
    }
    // Usamos o opencv
    else {
	    mask_img = cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);

      if(mask_img.empty()) {
          printf("Falha ao ler a máscara\n");
          exit(1);
      }

      // Alocamos as imagens
      cudaMallocPitch(&d_mask, &pitchMask, size*sizeof(unsigned char), size);

      // Enviamos a imagem para a GPU
      cudaMemcpy2D(d_mask, pitchMask, mask_img.data, mask_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("Tempo GPU: %ld ms\n", ms_int.count());
    printf("size: %d\n", size);

    cv::Mat output_img(cv::Size(size, size), CV_8UC1);
    // Enviamos o resultado para o host
    // cudaMemcpy2D(output_img.data, output_img.step, 
    //                 d_marker, pitchMarker, 
    //                 size * sizeof(unsigned char), size, 
    //                 cudaMemcpyDeviceToHost);

    // cv::imwrite(output_filename, output_img);

    cudaFree(d_marker);
    //cudaFree(d_mask);
    
    return 0;
}
