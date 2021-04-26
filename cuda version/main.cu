#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PLATEAU 0
#define BLOCK_SIZE 16

// Convert 2D index to 1D index.
#define INDEX(j, i, ld) ((j)*ld + (i))

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind, off) (((ind) / BLOCK_SIZE) * (BLOCK_SIZE - 2) - 1 + (off))

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans)                           \
    {                                                 \
        gpuAssert((ans), (char *)__FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__constant__ int neighbour_x[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
__constant__ int neighbour_y[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

typedef unsigned char image_t, *image_ptr_t;
typedef int img_t, *img_ptr_t;

texture<img_t, 2, cudaReadModeElementType> image;

img_ptr_t convert2data(image_ptr_t image, const int width, const int height);
image_ptr_t convert2image(img_ptr_t image, const int width, const int height);
__global__ void steepest_descent_kernel(img_ptr_t in_out, const int width, const int height);
__global__ void increment_kernel(img_ptr_t in_out, const int width, const int height);
__global__ void border_kernel(img_ptr_t in_out, int *count, const int width, const int height);
__global__ void minima_basin_kernel(img_ptr_t in_out, int *count, const int width, const int height);
__global__ void watershed_kernel(img_ptr_t in_out, int *count, const int width, const int height);
double interval(struct timespec start, struct timespec end);
int main(int argc, char **argv);

int main(int argc, char **argv)
{
    int width, height, channels;
    image_ptr_t data = stbi_load(argv[1], &width, &height, &channels, 1);
    img_ptr_t input = convert2data(data, width, height);
    stbi_image_free(data);
    img_ptr_t cpu_lowest_descent = (img_ptr_t)calloc(width * height, sizeof(img_t));
    img_ptr_t cpu_border = (img_ptr_t)calloc(width * height, sizeof(img_t));
    img_ptr_t cpu_minima = (img_ptr_t)calloc(width * height, sizeof(img_t));
    img_ptr_t cpu_watershed = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (cpu_border == NULL || cpu_lowest_descent == NULL || cpu_minima == NULL || cpu_watershed == NULL)
    {
        fprintf(stderr, "Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    img_ptr_t gpu_memory;
    size_t offset = 0;
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaMalloc((img_ptr_t *)&gpu_memory, width * height * sizeof(img_t)));
    CUDA_SAFE_CALL(cudaMalloc((img_ptr_t *)&image, width * height * sizeof(img_t)));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_memory, input, width * height * sizeof(img_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaBindTexture(&offset, image, gpu_memory, width * height * sizeof(img_t)));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(width / (threadsPerBlock.x - 2), height / (threadsPerBlock.y - 2), 1);
    int *count = NULL;
    int _old = -1;
    int _new = -2;
    CUDA_SAFE_CALL(cudaMallocManaged((int **)&count, sizeof(int)));
    steepest_descent_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_memory, width, height);
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaMemcpy(cpu_lowest_descent, gpu_memory, width * height * sizeof(img_t), cudaMemcpyDeviceToHost));
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_memory, width, height);
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    *count = 0;
    while (_old != _new)
    {
        _old = _new;
        border_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_memory, count, width, height);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        _new = *count;
    }
    CUDA_SAFE_CALL(cudaMemcpy(cpu_border, gpu_memory, width * height * sizeof(img_t), cudaMemcpyDeviceToHost));
    *count = 0;
    _old = -1;
    _new = -2;
    while (_old != _new)
    {
        _old = _new;
        minima_basin_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_memory, count, width, height);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        _new = *count;
    }
    CUDA_SAFE_CALL(cudaMemcpy(cpu_minima, gpu_memory, width * height * sizeof(img_t), cudaMemcpyDeviceToHost));
    *count = 0;
    _old = -1;
    _new = -2;
    while (_old != _new)
    {
        _old = _new;
        watershed_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_memory, count, width, height);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        _new = *count;
    }
    CUDA_SAFE_CALL(cudaMemcpy(cpu_watershed, gpu_memory, width * height * sizeof(img_t), cudaMemcpyDeviceToHost));
    stbi_write_png("1_lowest_descent_result.png", width, height, channels, convert2image(cpu_lowest_descent, width, height), width * channels);
    stbi_write_png("2_border_result.png", width, height, channels, convert2image(cpu_border, width, height), width * channels);
    stbi_write_png("3_minima_basin_result.png", width, height, channels, convert2image(cpu_minima, width, height), width * channels);
    stbi_write_png("4_watershed_result.png", width, height, channels, convert2image(cpu_watershed, width, height), width * channels);
    CUDA_SAFE_CALL(cudaDeviceReset());
    return 0;
}

double interval(struct timespec start, struct timespec end)
{
    /*
    This method does not require adjusting a #define constant

    How to use this method:

        struct timespec time_start, time_stop;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        // DO SOMETHING THAT TAKES TIME
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        measurement = interval(time_start, time_stop);*/
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}

img_ptr_t convert2data(image_ptr_t image, const int width, const int height)
{
    img_ptr_t temp = (img_ptr_t)calloc(width * height, sizeof(img_t));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            temp[i * width + j] = (img_t)image[i * width + j];
        }
    }
    return temp;
}

image_ptr_t convert2image(img_ptr_t image, const int width, const int height)
{
    // Step 1: find min and max values from the image
    img_t max = INT_MIN, min = INT_MAX;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            img_t current_pixel = image[i * width + j];
            if (current_pixel < min)
                min = current_pixel;
            if (current_pixel > max)
                max = current_pixel;
        }

    // Step 2: create a new image with the values scaled from [0-255]
    image_ptr_t temp = (image_ptr_t)calloc(width * height, sizeof(image_t));
    float max_min = max - min;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            img_t pix_val = image[i * width + j];
            float val = (pix_val - min) / max_min;
            temp[i * width + j] = (image_t)(val * 255);
        }
    }
    return temp;
}

__global__ void increment_kernel(img_ptr_t in_out, const int width, const int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int p = INDEX(j, i, width);

    if (j < height && i < width && in_out[p] == PLATEAU)
    {
        in_out[p] += 1;
    }
}

__global__ void steepest_descent_kernel(img_ptr_t in_out, const int width, const int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int i = bdx * bx + tx;
    int j = bdy * by + ty;

    __shared__ float s_I[BLOCK_SIZE * BLOCK_SIZE];
    int size = BLOCK_SIZE - 2;
    int img_x = L2I(i, tx);
    int img_y = L2I(j, ty);
    int new_w = width + width * 2;
    int new_h = height + height * 2;
    int p = INDEX(img_y, img_x, width);

    int ghost = (tx == 0 || ty == 0 ||
                 tx == bdx - 1 || ty == bdy - 1);

    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
        (bx == (width / size - 1) && tx == bdx - 1) ||
        (by == (height / size - 1) && ty == bdy - 1))
    {
        s_I[INDEX(ty, tx, BLOCK_SIZE)] = INFINITY;
    }
    else
    {
        s_I[INDEX(ty, tx, BLOCK_SIZE)] = tex2D(image, img_x, img_y);
    }

    __syncthreads();

    if (j < new_h && i < new_w && ghost == 0)
    {
        float I_q_min = INFINITY;
        float I_p = tex2D(image, img_x, img_y);

        int exists_q = 0;

        for (int k = 0; k < 8; k++)
        {
            int n_x = neighbour_x[k] + tx;
            int n_y = neighbour_y[k] + ty;
            float I_q = s_I[INDEX(n_y, n_x, BLOCK_SIZE)];
            if (I_q < I_q_min)
                I_q_min = I_q;
        }

        for (int k = 0; k < 8; k++)
        {
            int x = neighbour_x[k];
            int y = neighbour_y[k];
            int n_x = x + tx;
            int n_y = y + ty;
            int n_tx = L2I(i, n_x);
            int n_ty = L2I(j, n_y);
            float I_q = s_I[INDEX(n_y, n_x, BLOCK_SIZE)];
            int q = INDEX(n_ty, n_tx, width);
            if (I_q < I_p && I_q == I_q_min)
            {
                in_out[p] = -q;
                exists_q = 1;
                break;
            }
        }
        if (exists_q == 0)
            in_out[p] = PLATEAU;
    }
}

__global__ void border_kernel(img_ptr_t in_out, int *count, const int width, const int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int i = bdx * bx + tx;
    int j = bdy * by + ty;

    __shared__ float s_L[BLOCK_SIZE * BLOCK_SIZE];
    int size = BLOCK_SIZE - 2;
    int img_x = L2I(i, tx);
    int img_y = L2I(j, ty);
    int true_p = INDEX(img_y, img_x, width);
    int s_p = INDEX(ty, tx, BLOCK_SIZE);
    int new_w = width + width * 2;
    int new_h = height + height * 2;
    int ghost = (tx == 0 || ty == 0 ||
                 tx == bdx - 1 || ty == bdy - 1)
                    ? 1
                    : 0;

    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
        (bx == (width / size - 1) && tx == bdx - 1) ||
        (by == (height / size - 1) && ty == bdy - 1))
    {
        s_L[INDEX(ty, tx, BLOCK_SIZE)] = INFINITY;
    }
    else
    {
        s_L[s_p] = in_out[INDEX(img_y, img_x, width)];
    }

    __syncthreads();

    int active = (j < new_h && i < new_w && s_L[s_p] > 0) ? 1 : 0;

    if (active == 1 && ghost == 0)
    {
        for (int k = 0; k < 8; k++)
        {
            int n_x = neighbour_x[k] + tx;
            int n_y = neighbour_y[k] + ty;
            int s_q = INDEX(n_y, n_x, BLOCK_SIZE);
            if (s_L[s_q] == INFINITY)
                continue;
            if (s_L[s_q] > s_L[s_p])
                s_L[s_p] = s_L[s_q];
        }
        if (in_out[true_p] != s_L[s_p])
        {
            in_out[true_p] = s_L[s_p];
            atomicAdd(count, 1);
        }
    }
}

__global__ void minima_basin_kernel(img_ptr_t in_out, int *count, const int width, const int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int i = bdx * bx + tx;
    int j = bdy * by + ty;

    __shared__ float s_L[BLOCK_SIZE * BLOCK_SIZE];
    int size = BLOCK_SIZE - 2;
    int img_x = L2I(i, tx);
    int img_y = L2I(j, ty);
    int true_p = INDEX(img_y, img_x, width);
    int p = INDEX(ty, tx, BLOCK_SIZE);
    int new_w = width + width * 2;
    int new_h = height + height * 2;
    int ghost = (tx == 0 || ty == 0 ||
                 tx == bdx - 1 || ty == bdy - 1);

    // Load data into shared memory.
    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
        (bx == (width / size - 1) && tx == bdx - 1) ||
        (by == (height / size - 1) && ty == bdy - 1))
    {
        s_L[INDEX(ty, tx, BLOCK_SIZE)] = INFINITY;
    }
    else
    {
        s_L[INDEX(ty, tx, BLOCK_SIZE)] = in_out[INDEX(img_y, img_x, width)];
    }

    __syncthreads();

    if (j < new_h && i < new_w &&
        s_L[p] == PLATEAU && ghost == 0)
    {
        float I_p = tex2D(image, img_x, img_y);
        float I_q;
        int n_x, n_y;
        float L_q;

        for (int k = 0; k < 8; k++)
        {
            n_x = neighbour_x[k] + tx;
            n_y = neighbour_y[k] + ty;
            L_q = s_L[INDEX(n_y, n_x, BLOCK_SIZE)];
            if (L_q == INFINITY || L_q >= 0)
                continue;
            int n_tx = L2I(i, n_x);
            int n_ty = L2I(j, n_y);
            int q = INDEX(n_ty, n_tx, width);
            I_q = tex2D(image, n_tx, n_ty);
            if (I_q == I_p && in_out[true_p] != -q)
            {
                in_out[true_p] = -q;
                atomicAdd(count, 1);
                break;
            }
        }
    }
}

__global__ void watershed_kernel(img_ptr_t in_out, int *count, const int width, const int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int p = INDEX(j, i, width);
    int q;

    if (j < height && i < width && in_out[p] <= 0)
    {
        q = -in_out[p];
        if (in_out[q] > 0 && in_out[p] != in_out[q])
        {
            in_out[p] = in_out[q];
            atomicAdd(count, 1);
        }
    }
}