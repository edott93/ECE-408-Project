#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1 // 1 input
#define NUM_DIGITS 10

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS}; //actual input data dimensions
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS}; //reference labels. Each sample in batch contains a vector of NUM_DIGITS size. 

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32}; //5 x 5 filter. 1 input channel featured, 32 output features
static int conv2dims[] = {5, 5, 32, 64}; //32 input features, 64 output features
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// From book chapter Figure 16.4

// static void conv_forward_valid(const float *X, const int xdims[4],
//                                const float *W, const int wdims[4], float *Y,
//                                const int ydims[4]) {
//   const auto filter_h   = wdims[0];
//   const auto filter_w   = wdims[1];
//   const auto in_channel = wdims[2];

//   for (const auto i : range(0, ydims[0])) { //for each sample in batch 
//     for (const auto m : range(0, ydims[3])) { //for each output feature map
//       for (const auto w : range(0, ydims[2])) { //for width of output element
//         for (const auto h : range(0, ydims[1])) { //for height of output element
//           for (const auto p : range(0, filter_h)) { //for height of filter
//             for (const auto q : range(0, filter_w)) { //for width of filter
//               for (const auto c : range(0, in_channel)) { //for each input feature
//                 const auto yoffset =
//                     ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
//                 const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
//                                      (h + p) * xdims[2] * xdims[3] +
//                                      (w + q) * xdims[3] + c;
//                 const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
//                                      q * wdims[2] * wdims[3] + c * wdims[3] + m;
//                 Y[yoffset] += X[xoffset] * W[woffset];
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }



// From book chapter Figure 16.5
// static void average_pool(const float *X, const int xdims[4],
//                          const int pool_size, float *Y, const int ydims[4]) {
//   for (const auto i : range(0, ydims[0])) {
//     for (const auto m : range(0, ydims[3])) {
//       for (const auto w : range(0, ydims[2])) {
//         for (const auto h : range(0, ydims[1])) {
//           for (const auto p : range(0, pool_size)) {
//             for (const auto q : range(0, pool_size)) {
//               const auto yoffset =
//                   ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
//               const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
//                                    (pool_size * h + p) * xdims[2] * xdims[3] +
//                                    (pool_size * w + q) * xdims[3] + m;
//               Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
//             }
//           }
//         }
//       }
//     }
//   }
// }


// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}


// __global__ void relu(float *deviceX, int size)
// {
//   int t =  blockIdx.x * 1024 + threadIdx.x;
//   if (t < size)
//     deviceX[t] = (deviceX[t] < 0) ? 0 : deviceX[t];
// }

// void relu2(float *X, const int xdims[2]) 
// {
//   float * deviceX;
//   int size = xdims[0] * xdims[1];

//   cudaMalloc((void**) &deviceX, size * sizeof(float));
//   cudaMemcpy(deviceX, X, size * sizeof(float), cudaMemcpyHostToDevice);


//   dim3 DimBlock(1024, 1, 1);

//   int num_threadsInput = size;
//   int num_blocksInput =  ceil((num_threadsInput + 1023) / 1024);    
//   dim3 DimGrid(num_blocksInput, 1, 1);

//   relu<<<DimGrid, DimBlock>>> (deviceX, size);
//   cudaMemcpy(X, deviceX, size * sizeof(float), cudaMemcpyDeviceToHost);   


// }

// void relu4(float *X, const int xdims[4]) 
// {
//   float * deviceX;
//   int size = xdims[0] * xdims[1] * xdims[2] * xdims[3];

//   cudaMalloc((void**) &deviceX, size * sizeof(float));
//   cudaMemcpy(deviceX, X, size * sizeof(float), cudaMemcpyHostToDevice);


//   dim3 DimBlock(1024, 1, 1);

//   int num_threadsInput = size;
//   int num_blocksInput =  ceil((num_threadsInput + 1023) / 1024);    
//   dim3 DimGrid(num_blocksInput, 1, 1);

//   relu<<<DimGrid, DimBlock>>> (deviceX, size);

//   cudaMemcpy(X, deviceX, size * sizeof(float), cudaMemcpyDeviceToHost);   
// }


__global__ void unroll_InputOptimized(int C, int H_out, int W_out, int K, int W, float *X, float *X_unroll, int index)
{
    int t =  blockIdx.x * 1024 + threadIdx.x;
    int W_unroll = H_out * W_out;
    int c, s, h_out, w_out, h_unroll, w_unroll, w_base;
    if (t < C * W_unroll)
    {
      c = t % C;
      s = t / C;
      //printf(" %d", s);
      h_out = s / W_out;
      w_out = s % W_out;
      h_unroll = s;
      w_base = c * K * K;
      for (const p : range(0, K))
      {
        for (const q : range(0, K))
        {
          w_unroll = w_base + p * K + q;   
          X_unroll[w_unroll * H_out * W_out + h_unroll] = X[index + (h_out + p) * W * C + (w_out + q) * C + c];
          //printf("%d ", (h_out + p) * W * C + (w_out + q) * C + c);
        }
      }


    }
}

__global__ void unroll_W(int C, int M, int K, float * W, float * W_unroll)
{
    //C 3, M 2, K 2
    int t = blockIdx.x * 1024 + threadIdx.x;
    if (t < C * M)
    {
      int m = t % M;
      int c = t / M;
      int unroll_width = C * K * K;
      for (const p : range(0, K))
      {
        for (const q : range(0, K))
        {
          W_unroll[unroll_width * m + K * K * c + p * K + q] = W[p * K * C * M + q * C * M + c * M + m];
          //printf("%d ", p * K * C * M + q * C * M + c * M + m);
        }
      }
    }
}

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  __shared__ float subTileM[16][16];
  __shared__ float subTileN[16][16];
  
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  
  int Row = by * 16 + ty;  
  int Col = bx * 16 + tx;  
  float Pvalue = 0;
  
  for (int m = 0; m < ((numAColumns + 16) - 1)/16; m++)
  {
      if (m * 16 + tx < numAColumns && Row < numARows) 
      {
        subTileM[ty][tx] = A[Row*numAColumns+m*16+tx];
      }
      else 
      {
        subTileM[ty][tx] = 0;
      }
      if (m * 16 + ty < numBRows && Col < numBColumns)
      {
        subTileN[ty][tx] = B[(m*16+ty)*numBColumns + Col];
      }
      else
      {
      subTileN[ty][tx] = 0;
      }
    
      __syncthreads();
      for(int k = 0; k < 16; k++)
       {
          Pvalue += subTileM[ty][k] * subTileN[k][tx];
         __syncthreads();
       }
      if (Row < numCRows && Col < numCColumns)
      {
          C[Row * numCColumns + Col] = Pvalue;
      }
    
  }
}

__global__ void matrixMultiply1(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  __shared__ float subTileM[16][16];
  __shared__ float subTileN[16][16];
  
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  
  int Row = by * 16 + ty;  
  int Col = bx * 16 + tx;  
  float Pvalue = 0;
  
  for (int m = 0; m < ((numAColumns + 16) - 1)/16; m++)
  {
      if (m * 16 + tx < numAColumns && Row < numARows) 
      {
        subTileM[ty][tx] = A[Row*numAColumns+m*16+tx];
      }
      else 
      {
        subTileM[ty][tx] = 0;
      }
      if (m * 16 + ty < numBRows && Col < numBColumns)
      {
        subTileN[ty][tx] = B[(m*16+ty)*numBColumns + Col];
      }
      else
      {
      subTileN[ty][tx] = 0;
      }
    
      __syncthreads();
      for(int k = 0; k < 16; k++)
       {
          Pvalue += subTileM[ty][k] * subTileN[k][tx];
         __syncthreads();
       }
      if (Row < numCRows && Col < numCColumns)
      {
          if (Pvalue < 0)
            C[Row * numCColumns + Col] = 0;
          else C[Row * numCColumns + Col] = Pvalue;
      }
    
  }
}

__global__ void placeIntoY(float *Y_unroll, float *deviceY, int H, int W, int index)
{
    int t =  blockIdx.x * 1024 + threadIdx.x;
    if (t < W * H)
    {
      int h = t / W;
      int w = t % W;

      if (Y_unroll[w + h * W] < 0)
      {
        deviceY[index + w * H + h] = 0;
      }
      else deviceY[index + w * H + h] = Y_unroll[w + h * W];
      
     
    }

}

void convLayer_forward(int xdims[4], int wdims[4], float* X, float* Y, float* W)
{

    float *deviceX;
    float *deviceX2;

    float *deviceW;
    float *deviceW2;

    float *deviceY;
    float *deviceY2;

    //float *Xcheck;

    //float *deviceYUnrollCheck;
    //float *deviceXUnrollCheck;
    //float *deviceWUnrollCheck;

    float *deviceUnrollX;
    float *deviceUnrollX2;

    float *deviceUnrollW;

    float *deviceUnrollY;
    float *deviceUnrollY2;

    int H_out = xdims[1] - wdims[0] + 1;
    int W_out = xdims[2] - wdims[1] + 1;

    int M = wdims[3];
    int C = wdims[2];
    int N = xdims[0];

    //cudaMalloc((void**) &deviceX, xdims[1] * xdims[2] * xdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceX, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float));

    cudaMalloc((void**) &deviceW, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceY, xdims[0] * (xdims[1] - wdims[0] + 1) * (xdims[2] - wdims[1] + 1) *  wdims[3] * sizeof(float));
    //Xcheck = (float *)malloc(xdims[1] * xdims[2] * xdims[3] * sizeof(float));


    cudaMalloc((void**) &deviceX2, xdims[1] * xdims[2] * xdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceW2, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceY2, xdims[0] * (xdims[1] - wdims[0] + 1) * (xdims[2] - wdims[1] + 1) *  wdims[3] * sizeof(float));
    

    cudaMalloc((void**) &deviceUnrollY, M * H_out * W_out * sizeof(float));
    cudaMalloc((void**) &deviceUnrollX, H_out * W_out * (wdims[0] * wdims[1] * C) * sizeof(float));
    cudaMalloc((void**) &deviceUnrollW, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float));


    cudaMalloc((void**) &deviceUnrollY2, M * H_out * W_out * sizeof(float));
    cudaMalloc((void**) &deviceUnrollX2, H_out * W_out * (wdims[0] * wdims[1] * C) * sizeof(float));


    //deviceWUnrollCheck = (float *)malloc(wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float));
    //deviceYUnrollCheck = (float *)malloc(M * H_out * W_out * sizeof(float));
    //deviceXUnrollCheck = (float *)malloc(H_out * W_out * (wdims[0] * wdims[1] * C) * sizeof(float));

    
    int X_height = C * wdims[0] * wdims[1];
    int X_width = H_out * W_out;

    int W_width = X_height;
    int W_height = M;
    

    int Y_height = W_height;
    int Y_width = X_width;
    

    dim3 DimBlock(1024, 1, 1);

    int num_threadsInput = C * H_out * W_out;
    int num_blocksInput =  ceil((num_threadsInput + 1023) / 1024);
    //printf("numblocksinput: %d", num_blocksInput);
    dim3 DimGridInput(num_blocksInput, 1, 1);

    int num_threadsW = C * M;
    //printf("num_threadsW is %d", num_threadsW);
    int num_blocksW = ((num_threadsW + 1023) / 1024);
    //printf("num_blocksW is %d", num_blocksW);
    dim3 DimGridW(num_blocksW, 1, 1);

    dim3 DimBlockMultiply(16, 16, 1);
    int x = (Y_width+ 16 - 1)/16;
    int y = (Y_height + 16 - 1)/16;
    dim3 DimGridMultiply(x, y, 1);

    int num_threadsY = H_out * W_out * M;
    int num_blocksY = ((num_threadsY + 1023) / 1024);
    dim3 DimGridY(num_blocksY, 1, 1);

    cudaMemcpy(deviceW, W, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float), cudaMemcpyHostToDevice);
   
    /*
    for (const auto i : range(0, N))
    {
        cudaMemcpy(deviceX, &X[i * xdims[1] * xdims[2] * xdims[3]], xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice);

        unroll_InputOptimized<<<DimGridInput, DimBlock>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX, deviceUnrollX);

        cudaDeviceSynchronize();

        //cudaMemcpy(deviceXUnrollCheck, deviceUnrollX, H_out * W_out * (wdims[0] * wdims[1] * C) * sizeof(float), cudaMemcpyDeviceToHost);        

        unroll_W<<<DimGridW, DimBlock>>>(C, M, wdims[0], deviceW, deviceUnrollW);

        cudaDeviceSynchronize();
        //unroll_W<<<DimGridW, DimBlock>>>(3, 2, 2, deviceW, deviceUnrollW);


        //cudaMemcpy(deviceWUnrollCheck, deviceUnrollW, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float), cudaMemcpyDeviceToHost);        
        
        matrixMultiply<<<DimGridMultiply, DimBlockMultiply>>> (deviceUnrollW, deviceUnrollX, deviceUnrollY, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width);

        cudaDeviceSynchronize();

        cudaMemcpy(deviceYUnrollCheck, deviceUnrollY, M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);        


        placeIntoY<<<DimGridY, DimBlock>>>(deviceUnrollY, deviceY, M, H_out * W_out);

        cudaDeviceSynchronize();

        cudaMemcpy(&Y[i * H_out * W_out * M], deviceY, H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost);      
    } 
    */
    
    /*
    unroll_W<<<DimGridW, DimBlock>>>(C, M, wdims[0], deviceW, deviceUnrollW);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    for (auto i = 0; i < N; i+=2)
    {
        cudaMemcpyAsync(deviceX, &X[i * xdims[1] * xdims[2] * xdims[3]], xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(deviceX2, &X[(i+1) * xdims[1] * xdims[2] * xdims[3]], xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice, stream1);


        unroll_InputOptimized<<<DimGridInput, DimBlock, 0, stream0>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX, deviceUnrollX);
        unroll_InputOptimized<<<DimGridInput, DimBlock, 0, stream1>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX2, deviceUnrollX2);

        //cudaMemcpy(deviceXUnrollCheck, deviceUnrollX, H_out * W_out * (wdims[0] * wdims[1] * C) * sizeof(float), cudaMemcpyDeviceToHost);        
  
        //unroll_W<<<DimGridW, DimBlock>>>(3, 2, 2, deviceW, deviceUnrollW);

        //cudaMemcpy(deviceWUnrollCheck, deviceUnrollW, wdims[0] * wdims[1] * wdims[2] *  wdims[3] * sizeof(float), cudaMemcpyDeviceToHost);        
        
        matrixMultiply<<<DimGridMultiply, DimBlockMultiply, 0, stream0>>> (deviceUnrollW, deviceUnrollX, deviceUnrollY, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width);

        matrixMultiply<<<DimGridMultiply, DimBlockMultiply, 0, stream1>>> (deviceUnrollW, deviceUnrollX2, deviceUnrollY2, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width);


        //cudaMemcpy(deviceYUnrollCheck, deviceUnrollY, M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);        


        placeIntoY<<<DimGridY, DimBlock, 0, stream0>>>(deviceUnrollY, deviceY, M, H_out * W_out); 
        placeIntoY<<<DimGridY, DimBlock, 0, stream1>>>(deviceUnrollY2, deviceY2, M, H_out * W_out);

        cudaMemcpyAsync(&Y[i * H_out * W_out * M], deviceY, H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost, stream0);   
        cudaMemcpyAsync(&Y[(i+1) * H_out * W_out * M], deviceY2, H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost, stream1);      
   
    } 
    */

    
    unroll_W<<<DimGridW, DimBlock>>>(C, M, wdims[0], deviceW, deviceUnrollW);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaMemcpy(deviceX, X, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice);


    for (auto i = 0; i < N; i+=2)
    {
        int index = i * xdims[1] * xdims[2] * xdims[3];
        int index1 = (i+1) * xdims[1] * xdims[2] * xdims[3];

        int yindex = i * H_out * W_out * M;
        int yindex1 = (i+1) * H_out * W_out * M;

        unroll_InputOptimized<<<DimGridInput, DimBlock, 0, stream0>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX, deviceUnrollX, index);
        unroll_InputOptimized<<<DimGridInput, DimBlock, 0, stream1>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX, deviceUnrollX2, index1);    
        
        matrixMultiply<<<DimGridMultiply, DimBlockMultiply, 0, stream0>>> (deviceUnrollW, deviceUnrollX, deviceUnrollY, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width);

        matrixMultiply<<<DimGridMultiply, DimBlockMultiply, 0, stream1>>> (deviceUnrollW, deviceUnrollX2, deviceUnrollY2, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width);     

        placeIntoY<<<DimGridY, DimBlock, 0, stream0>>>(deviceUnrollY, deviceY, M, H_out * W_out, yindex);
        placeIntoY<<<DimGridY, DimBlock, 0, stream1>>>(deviceUnrollY2, deviceY, M, H_out * W_out, yindex1); 
   
    } 

     cudaMemcpy(Y, deviceY, N * H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost);   
    
/*
     unroll_W<<<DimGridW, DimBlock>>>(C, M, wdims[0], deviceW, deviceUnrollW);

    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaMemcpy(deviceX, X, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice);

        int inputsize = xdims[1] * xdims[2] * xdims[3];

        int outputsize = H_out * W_out * M;

        unroll_InputOptimized<<<DimGridInput, DimBlock>>>(C, H_out, W_out, wdims[0], xdims[2], deviceX, deviceUnrollX, inputsize);
        
        matrixMultiply<<<DimGridMultiply, DimBlockMultiply>>> (deviceUnrollW, deviceUnrollX, deviceUnrollY, W_height, W_width, X_height, 
                                        X_width, Y_height, Y_width); 

        placeIntoY<<<DimGridY, DimBlock, 0, stream0>>>(deviceUnrollY, deviceY, M, H_out * W_out, outputsize);

     cudaMemcpy(Y, deviceY, N * H_out * W_out * M * sizeof(float), cudaMemcpyDeviceToHost);   
*/
 
}

/*
__global__ void subsample(float *deviceInput, int inputH, int inputW, int outputW, int outputH, float *deviceOutput, int poolsize, int M, int index, int outdex)
{
  int t =  blockIdx.x * 1024 + threadIdx.x;
  if (t < (outputH * outputW * M))
  {
    int m = t % M;
    //int m = t % 3;

    int distance = t / M * poolsize;
    //int distance = t / 3 * poolsize;

    int w = distance % inputW; 
    //int w = distance % 4;

    int h = (distance / inputW) * poolsize;
    //int h = (distance / 4) * poolsize;

    float sum = 0;
    float average = 0;
    for (const p : range(0, poolsize))
      {
        for (const q : range(0, poolsize))
        {
          sum += deviceInput[index + (h + p) * inputH * M + (w + q) * M + m];
        }
      }

      average = sum / float(poolsize * poolsize);
      deviceOutput[outdex + (h/poolsize) * outputH * M + (w/poolsize) * M + m] = average;
  }
}
*/

__global__ void subsample(float *deviceInput, int inputH, int inputW, int outputW, int outputH, float *deviceOutput, int poolsize, int M, 
  int inputsize, int outputsize, int N)
{
  int t =  blockIdx.x * 1024 + threadIdx.x;
  if (t < (outputH * outputW * M * N))
  {

    int index = t/(outputH * outputW * M);

    int m = (t%(outputH * outputW * M)) % M;
    //int m = t % 3;

    int distance = (t%(outputH * outputW * M)) / M * poolsize;
    //int distance = t / 3 * poolsize;

    int w = distance % inputW; 
    //int w = distance % 4;

    int h = (distance / inputW) * poolsize;
    //int h = (distance / 4) * poolsize;

    float sum = 0;
    float average = 0;
    for (const p : range(0, poolsize))
      {
        for (const q : range(0, poolsize))
        {
          sum += deviceInput[(index * inputsize) + (h + p) * inputH * M + (w + q) * M + m];
        }
      }

      average = sum / float(poolsize * poolsize);
      deviceOutput[(index * outputsize) + (h/poolsize) * outputH * M + (w/poolsize) * M + m] = average;
      //__syncthreads();


  }
}

void subsampling_layer(float *input, float *output, int poolsize, int inputdims[4], int outputdims[4])
{
    float * deviceX;
    float * deviceY;

    float * deviceX2;
    float * deviceY2;

    int N = inputdims[0];
    int H_in = inputdims[1];
    int W_in = inputdims[2];

    int H_out = outputdims[1];
    int W_out = outputdims[2];


    cudaMalloc((void**) &deviceX, inputdims[0] * inputdims[1] * inputdims[2] * inputdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceY, outputdims[0] * outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float));

    cudaMalloc((void**) &deviceX2, inputdims[1] * inputdims[2] * inputdims[3] * sizeof(float));
    cudaMalloc((void**) &deviceY2, outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float));

    dim3 DimBlockSample(1024, 1, 1);

    int num_threadsInput = outputdims[1] * outputdims[2] * outputdims[3] * N;
    int num_blocksInput =  ceil((num_threadsInput + 1023) / 1024);
    //printf("numblocksinput: %d", num_blocksInput);
    dim3 DimGridSample(num_blocksInput, 1, 1);


    cudaStream_t stream0;
    cudaStream_t stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    /*
    for (const auto i : range(0, N))
    {
       cudaMemcpy(deviceX, &input[i * inputdims[1] * inputdims[2] * inputdims[3]], inputdims[1] * 
        inputdims[2] * inputdims[3] * sizeof(float), cudaMemcpyHostToDevice);

       subsample<<<DimGridSample, DimBlockSample>>>(deviceX, H_in, W_in, W_out, H_out, deviceY, poolsize, inputdims[3]);
       cudaDeviceSynchronize();

        cudaMemcpy(&output[i * outputdims[1] * outputdims[2] * outputdims[3]], deviceY, 
          outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float), cudaMemcpyDeviceToHost);      
    }

    */
    
/*
    
    for (auto i = 0; i < N; i+=2)
    {
       cudaMemcpyAsync(deviceX, &input[i * inputdims[1] * inputdims[2] * inputdims[3]], inputdims[1] * 
        inputdims[2] * inputdims[3] * sizeof(float), cudaMemcpyHostToDevice, stream0);

       cudaMemcpyAsync(deviceX2, &input[(i+1) * inputdims[1] * inputdims[2] * inputdims[3]], inputdims[1] * 
        inputdims[2] * inputdims[3] * sizeof(float), cudaMemcpyHostToDevice, stream1);

       subsample<<<DimGridSample, DimBlockSample, 0, stream0>>>(deviceX, H_in, W_in, W_out, H_out, deviceY, poolsize, inputdims[3]);

       subsample<<<DimGridSample, DimBlockSample, 0, stream1>>>(deviceX2, H_in, W_in, W_out, H_out, deviceY2, poolsize, inputdims[3]);

       cudaMemcpyAsync(&output[i * outputdims[1] * outputdims[2] * outputdims[3]], deviceY, 
       outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float), cudaMemcpyDeviceToHost, stream0);   

       cudaMemcpyAsync(&output[(i+1) * outputdims[1] * outputdims[2] * outputdims[3]], deviceY2, 
       outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float), cudaMemcpyDeviceToHost, stream1);    

    }

    */

    /*

     cudaMemcpy(deviceX, input, inputdims[0] * inputdims[1] * 
        inputdims[2] * inputdims[3] * sizeof(float), cudaMemcpyHostToDevice);

    for (auto i = 0; i < N; i+=2)
    {

      int index = i * inputdims[1] * inputdims[2] * inputdims[3];
      int index1 = (i + 1) * inputdims[1] * inputdims[2] * inputdims[3];

      int outindex =  i * outputdims[1] * outputdims[2] * outputdims[3];
      int outindex1 =  (i+1)* outputdims[1] * outputdims[2] * outputdims[3];



       subsample<<<DimGridSample, DimBlockSample, 0, stream0>>>(deviceX, H_in, W_in, W_out, H_out, deviceY, poolsize, inputdims[3], index, outindex);

       subsample<<<DimGridSample, DimBlockSample, 0, stream1>>>(deviceX, H_in, W_in, W_out, H_out, deviceY, poolsize, inputdims[3], index1, outindex1);  

    }
    
    cudaMemcpy(output, deviceY, outputdims[0] * outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float), cudaMemcpyDeviceToHost);   

    */
    
    cudaMemcpy(deviceX, input, inputdims[0] * inputdims[1] * 
        inputdims[2] * inputdims[3] * sizeof(float), cudaMemcpyHostToDevice);
    
    int inputsize = inputdims[1] * inputdims[2] * inputdims[3];
    int outputsize = outputdims[1] * outputdims[2] * outputdims[3];


    subsample<<<DimGridSample, DimBlockSample>>>(deviceX, H_in, W_in, W_out, H_out, deviceY, poolsize, inputdims[3], inputsize, outputsize, N);


    
    cudaMemcpy(output, deviceY, outputdims[0] * outputdims[1] * outputdims[2] * outputdims[3] * sizeof(float), cudaMemcpyDeviceToHost);   
}

void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2], int check) {

  int numARows = xdims[0], numAColumns = xdims[1];
  int numBRows = wdims[0], numBColumns = wdims[1];
  //int numCRows = numARows, numCColumns = numBColumns;
  int numCRows = ydims[0], numCColumns = ydims[1];
  
  float *deviceA;
  float *deviceB;
  float *deviceC;

  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void**) &deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void**) &deviceC, numCRows*numCColumns*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, X, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, W, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/16.0), ceil(numCRows / 16.0), 1);
  if(numCColumns % 16) {
    DimGrid.x++;
  }
  if(numCRows % 16) {
    DimGrid.y++;
  }
  dim3 DimBlock(16, 16, 1);

  //@@ Launch the GPU Kernel here
  if (check == 1)
    matrixMultiply1<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,
                               numAColumns, numBRows,
                               numBColumns, numCRows,
                               numCColumns);
  else matrixMultiply<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,
                               numAColumns, numBRows,
                               numBColumns, numCRows,
                               numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(Y, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {

  // conv layer
  int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]}; //batch size, output_height, output_width, number of output features
  auto a = zeros<float>(adims);
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  convLayer_forward(xdims, conv1dims, x, a, conv1);


  /// relu layer
  //relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  //average_pool(a, adims, pool_size, b, bdims);
  subsampling_layer(a, b, pool_size, adims, bdims);

  // conv layer
  int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  //conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  convLayer_forward(bdims, conv2dims, b, c, conv2);


  // relu
  //relu4(c, cdims);

  // average pooling
  int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  //average_pool(c, cdims, pool_size, d, ddims);

  subsampling_layer(c, d, pool_size, cdims, ddims);


  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims, 1);

  // relu
  //relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims, 0);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

  //size_t sz = 1048576 * 4;
 //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims); //x contains data for images
  float *y = allocate<float>(rdims); //y contains label vectors of images
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform forward opertion
  int *out = zeros<int>(FLAGS_batch_size); //contains predicted labels for each sample

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size); //contains true labels for each sample
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;

  
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
