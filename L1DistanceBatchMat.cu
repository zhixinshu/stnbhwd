#include "utils.h"


// __device__ void sumReduceShMem(volatile float s[])


__global__ void pairwiseL1Distance(float* input_data, int input_strideBatch, int input_strideKernels, int input_strideElements, 
                                    int input_kernels, int input_sizeKernels,
                                    float* output_data, int output_strideBatch1, int output_strideBatch2)
{
   /* input is (bs, b, c)
      each block computes L1 for one pair of samples (blockIdx.x, blockIdx.y)
      samples are on the first dimension
   */

   float* vect1_data = input_data + blockIdx.x*input_strideBatch;
   float* vect2_data = input_data + blockIdx.y*input_strideBatch;

   __shared__ float accs[32];
   if(threadIdx.y==0 && threadIdx.x<32) accs[threadIdx.x]=0;

   for(int b=0; b<input_kernels; b++)
   {
      float acc = 0;
      for(int t=threadIdx.x; t<input_sizeKernels; t+=blockDim.x)
      {
         acc += abs(vect1_data[t] - vect2_data[t]);
      }
      accs[threadIdx.x]=acc;
      sumReduceShMem(accs);
      if(threadIdx.x==0 && threadIdx.y==0) 
      {
         output_data[blockIdx.x*output_strideBatch1+blockIdx.y*output_strideBatch2+b] = accs[0];
         output_data[blockIdx.x*output_strideBatch2+blockIdx.y*output_strideBatch1+b] = accs[0];
      }
      vect1_data += input_strideKernels;
      vect2_data += input_strideKernels;
   }
}

__global__ void pairwiseL1DistanceBackward(float* input_data, int input_strideBatch, int input_strideKernels, int input_strideElements,
                                           int input_kernels, int input_sizeKernels,
                                           float* gradInput_data, int gradInput_strideBatch, int gradInput_strideKernels,
                                           float* gradOutput_data, int gradOutput_strideBatch1, int gradOutput_strideBatch2)
{
   /* 
      each block does gradient for one pair, gradOutput[x][y]
      input is (bs, b, c)
      gradOutput is (bs, bs, b)

   */
   float* vect1_data = input_data + blockIdx.x*input_strideBatch;
   float* vect2_data = input_data + blockIdx.y*input_strideBatch;

   float* vect1gradInput_data = gradInput_data + blockIdx.x*gradInput_strideBatch;
   float* vect2gradInput_data = gradInput_data + blockIdx.y*gradInput_strideBatch;

   float* grad = gradOutput_data + blockIdx.x*gradOutput_strideBatch1 + blockIdx.y*gradOutput_strideBatch2; 

   for(int b=0; b<input_kernels; b++)
   {
      for(int t=threadIdx.x; t<input_sizeKernels; t+=blockDim.x)
      {
         float g = grad[b];
         float val1 = vect1_data[t];
         float val2 = vect2_data[t];
         if(val2>val1) {g=-g;}
         atomicAdd(&vect1gradInput_data[t], g);
         atomicAdd(&vect2gradInput_data[t], -g);
      }
      vect1_data += input_strideKernels;
      vect2_data += input_strideKernels;
      vect1gradInput_data += gradInput_strideKernels;
      vect2gradInput_data += gradInput_strideKernels;
   }
}

static int cunn_L1DistanceBatchMat_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   int bs = input->size[0];

   dim3 blocks(bs,bs);
   dim3 threads(32);

/*__global__ void pairwiseL1Distance(float* input_data, int input_strideBatch, int input_strideKernels, int input_strideElements, 
                                    int input_kernels, int input_sizeKernels,
                                    float* output_data, int output_strideBatch1, int output_strideBatch2)*/

   /* assume BHWD */
   pairwiseL1Distance <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, input), 
                                                      THCudaTensor_stride(state, input, 0), 
                                                      THCudaTensor_stride(state, input, 1), 
                                                      THCudaTensor_stride(state, input, 2),
                                                      THCudaTensor_size(state, input, 1), 
                                                      THCudaTensor_size(state, input, 2), 
                                                      THCudaTensor_data(state, output),  
                                                      THCudaTensor_stride(state, output, 0), 
                                                      THCudaTensor_stride(state, output, 1));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in PairwiseL1Distance.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}




static int cunn_L1DistanceBatchMat_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

   int bs = input->size[0];

   dim3 blocks(bs,bs);
   dim3 threads(32);

/* __global__ void pairwiseL1DistanceBackward(float* input_data, int input_strideBatch, int input_strideKernels, int input_strideElements,
                                           int input_kernels, int input_sizeKernels,
                                           float* gradInput_data, int gradInput_strideBatch, int gradInput_strideKernels,
                                           float* gradOutput_data, int gradOutput_strideBatch1, int gradOutput_strideBatch2)
*/

   pairwiseL1DistanceBackward <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, input), 
                                                      THCudaTensor_stride(state, input, 0), 
                                                      THCudaTensor_stride(state, input, 1), 
                                                      THCudaTensor_stride(state, input, 2),
                                                      THCudaTensor_size(state, input, 1), 
                                                      THCudaTensor_size(state, input, 2), 
                                                      THCudaTensor_data(state, gradInput),  
                                                      THCudaTensor_stride(state, gradInput, 0), 
                                                      THCudaTensor_stride(state, gradInput, 1),
                                                      THCudaTensor_data(state, gradOutput),  
                                                      THCudaTensor_stride(state, gradOutput, 0), 
                                                      THCudaTensor_stride(state, gradOutput, 1));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in PairwiseL1Distance.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}



static const struct luaL_Reg cunn_L1DistanceBatchMat__ [] = {
  {"L1DistanceBatchMat_updateOutput", cunn_L1DistanceBatchMat_updateOutput},
  {"L1DistanceBatchMat_updateGradInput", cunn_L1DistanceBatchMat_updateGradInput},
  {NULL, NULL}
};

static void cunn_L1DistanceBatchMat_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_L1DistanceBatchMat__, "nn");
  lua_pop(L,1);
}
