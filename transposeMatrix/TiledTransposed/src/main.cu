#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>


#define TILE_DIM 64
#define BLOCK_ROWS 8

#define DEVICE_MALLOC_LIMIT 1024*1024*1024


__global__ void transposeInPlaceTiled(unsigned char *data, int width, int height, size_t pitch)
{
  __shared__ unsigned char tileA[TILE_DIM][TILE_DIM + 1];
  __shared__ unsigned char tileB[TILE_DIM][TILE_DIM + 1];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Diagonal tile: transpose in-place
  if (bx == by) {
    // Load tile into shared memory

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + ty + i;
      int col = bx * TILE_DIM + tx*4;
      if (row < height && col < width) {
        uchar4 *rowPtr = (uchar4 *)((unsigned char*)data + row * pitch);
        uchar4 pixels = rowPtr[col / 4];
        tileA[ty + i][tx*4] = pixels.x;
        tileA[ty + i][tx*4+1] = pixels.y;
        tileA[ty + i][tx*4+2] = pixels.z;
        tileA[ty + i][tx*4+3] = pixels.w;
      }
    }


    __syncthreads();

    // Write transposed back

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int row = by * TILE_DIM + ty + i;
      int col = bx * TILE_DIM + tx*4;
      if (row < height && col < width) {
        uchar4 pixels_out;
        pixels_out.x = tileA[tx * 4][ty + i];
        pixels_out.y = tileA[tx * 4 + 1][ty + i];
        pixels_out.z = tileA[tx * 4 + 2][ty + i];
        pixels_out.w = tileA[tx * 4 + 3][ty + i];

        uchar4 *rowPtr_out = (uchar4 *)((unsigned char*)data + row * pitch);
        rowPtr_out[col / 4] = pixels_out;
      }
    }

  }
  // Off-diagonal tiles: swap tile (bx,by) with transposed tile (by,bx)
  else if (bx < by) {
    // Load A = tile(bx,by) and B = tile(by,bx)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + ty + i;
      int colA = bx * TILE_DIM + tx*4;
      if (rowA < height && colA < width) {
        uchar4 *rowPtrA = (uchar4 *)((char*)data + rowA * pitch);
        uchar4 pixels = rowPtrA[colA / 4];
        tileA[ty + i][tx*4] = pixels.x;
        tileA[ty + i][tx*4+1] = pixels.y;
        tileA[ty + i][tx*4+2] = pixels.z;
        tileA[ty + i][tx*4+3] = pixels.w;
      } else {
        tileA[ty + i][tx*4] = 0;
        tileA[ty + i][tx*4+1] = 0;
        tileA[ty + i][tx*4+2] = 0;
        tileA[ty + i][tx*4+3] = 0;
      }

      int rowB = bx * TILE_DIM + ty + i;
      int colB = by * TILE_DIM + tx*4;
      if (rowB < height && colB < width) {
        uchar4 *rowPtrB = (uchar4 *)((char*)data + rowB * pitch);
        uchar4 pixels = rowPtrB[colB / 4];
        tileB[ty + i][tx*4] = pixels.x;
        tileB[ty + i][tx*4+1] = pixels.y;
        tileB[ty + i][tx*4+2] = pixels.z;
        tileB[ty + i][tx*4+3] = pixels.w;
      } else {
        tileB[ty + i][tx*4] = 0;
        tileB[ty + i][tx*4+1] = 0;
        tileB[ty + i][tx*4+2] = 0;
        tileB[ty + i][tx*4+3] = 0;
      }
    }

    __syncthreads();

    // Write transposed: A <- transpose(B), B <- transpose(A)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int rowA = by * TILE_DIM + ty + i;
      int colA = bx * TILE_DIM + tx*4;
      // write into A region from tileB transposed
      if (rowA < height && colA < width) {
        uchar4 pixels_out;
        pixels_out.x = tileB[tx * 4][ty + i];
        pixels_out.y = tileB[tx * 4 + 1][ty + i];
        pixels_out.z = tileB[tx * 4 + 2][ty + i];
        pixels_out.w = tileB[tx * 4 + 3][ty + i];
        uchar4 *rowPtrA = (uchar4 *)((char*)data + rowA * pitch);
        rowPtrA[colA/4] = pixels_out;
      }

      int rowB = bx * TILE_DIM + ty + i;
      int colB = by * TILE_DIM + tx*4;
      // write into B region from tileA transposed
      if (rowB < height && colB < width) {
        uchar4 pixels_out;
        pixels_out.x = tileA[tx * 4][ty + i];
        pixels_out.y = tileA[tx * 4 + 1][ty + i];
        pixels_out.z = tileA[tx * 4 + 2][ty + i];
        pixels_out.w = tileA[tx * 4 + 3][ty + i];
        uchar4 *rowPtrB = (uchar4 *)((char*)data + rowB * pitch);
        rowPtrB[colB/4] = pixels_out;
      }
    }
  }
}


unsigned char* loadPGMToDevice(char* filename, int& width, int &height, size_t &pitch) {
  if(strcmp(&(filename[strlen(filename)-3]), "pgm")) {
    printf("Imagem inválida\n");
    exit(1);
  }

  int i;
  unsigned char* d_imagem_row;

  FILE* imagem;
  imagem = fopen(filename, "rb");
  if(imagem==NULL) {
    printf("Falha ao ler a imagem\n");
    exit(1);
  }

  char header_buffer[1024];

  char file_type[50];
  unsigned short max_val;

  memset(header_buffer, 0, 1024);
  fgets(header_buffer, 1024, imagem);
  sscanf(header_buffer, "%s\n", file_type);

  memset(header_buffer, 0, strlen(header_buffer));
  fgets(header_buffer, 1024, imagem);
  if(sscanf(header_buffer, "%d %d\n", &height, &width) < 2) {
    printf("Falha ao ler dimsnesões da imagem\n");
    exit(1);
  }

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

  uchar* buffer = (unsigned char*) malloc(width*sizeof(uchar)+32);

  for(i=0; i<height; i++) {
    if(fread(buffer, sizeof(unsigned char), width, imagem) < width) {
      printf("Falha na leitura (iteracao=%d)\n", i);
      exit(1);
    }
    
    d_imagem_row = (uchar*)((char*)d_imagem + i*pitch);
    err = cudaMemcpy(d_imagem_row, buffer, width*sizeof(unsigned char), cudaMemcpyHostToDevice);
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
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_LIMIT);
}

int main(int argc, char** argv) {
    
    if(argc < 3) {
      printf("Numero insuficiente de argumentos (%d/4)\n", argc-1);
      return 1;
    }

    char* marker_filename = argv[1];
    const char* output_filename = argv[2];
    int usePGMLoader = atoi(argv[3]);

    int size;
    int height;

    cv::Mat marker_img;
    
    unsigned char* d_marker;
    size_t pitchMarker;
    
    configureHeapSize();

    if(usePGMLoader) {
      d_marker = loadPGMToDevice(marker_filename, size, height, pitchMarker);
    }
    else {
      marker_img = cv::imread(marker_filename, cv::IMREAD_GRAYSCALE);
      
      if(marker_img.empty()) {
          printf("Falha ao ler o marcador\n");
          exit(1);
      }
      
      size = marker_img.size().width;
      cudaMallocPitch(&d_marker, &pitchMarker, size*sizeof(unsigned char), size);
      cudaMemcpy2D(d_marker, pitchMarker, marker_img.data, marker_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);

    }

    printf("size: %d\n", size);
    cudaDeviceSynchronize();

    // Using OpenCV to compute the transposed on CPU for correctness check
    if(usePGMLoader) {
      printf("Using custom PGM loader, skipping OpenCV transpose for correctness check\n");
      cv::Mat opencvTransposed;
      cv::transpose(marker_img, opencvTransposed);
    }

    
    auto tStart = std::chrono::high_resolution_clock::now();

    // Configure launch dimensions and run in-place transpose with coalesced reads
    dim3 dimBlock(TILE_DIM/4, BLOCK_ROWS);
    dim3 dimGrid((size + TILE_DIM - 1) / TILE_DIM, (size + TILE_DIM - 1) / TILE_DIM);
    transposeInPlaceTiled<<<dimGrid, dimBlock>>>(d_marker, size, size, pitchMarker);
    cudaError_t kernErr = cudaGetLastError();
    if (kernErr != cudaSuccess) {
      printf("Kernel launch error: %s\n", cudaGetErrorString(kernErr));
      return 1;
    }
    cudaDeviceSynchronize();

    auto tEnd = std::chrono::high_resolution_clock::now();

    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("Time to transpose: %ld ms\n", ms_int.count());
    
    
    if(!usePGMLoader){
      //  Only check when using OpenCV loader
      cv::Mat opencvTransposed;
      cv::transpose(marker_img, opencvTransposed);
      cv::Mat diff;
      // Difference between OpenCV and CUDA results
      cv::Mat output_img(cv::Size(size, size), CV_8UC1);
      cudaMemcpy2D(output_img.data, output_img.step, 
        d_marker, pitchMarker, 
        size * sizeof(unsigned char), size, 
        cudaMemcpyDeviceToHost);
      cv::absdiff(opencvTransposed, output_img, diff);
      cv::imwrite(output_filename, output_img);
      
      int erros = cv::countNonZero(diff);
      
      if (erros == 0) {
        printf("Success\n");
      } else {
        printf("Failed: %d pixels are different\n", erros);
      }
    }
    else{
      cv::Mat output_img(cv::Size(size, size), CV_8UC1);
      cudaMemcpy2D(output_img.data, output_img.step, 
        d_marker, pitchMarker, 
        size * sizeof(unsigned char), size, 
        cudaMemcpyDeviceToHost);
      //cv::imwrite(output_filename, output_img);
    }
    cudaFree(d_marker);
    
    return 0;
}
