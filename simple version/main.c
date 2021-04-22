#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PLATEAU 0

typedef unsigned char image_t, *image_ptr_t;
typedef float img_t, *img_ptr_t;

img_ptr_t convert2float(image_ptr_t image, int width, int height);
image_ptr_t convert2image(img_ptr_t image, int width, int height);
void lowest_descent_kernel(img_ptr_t in, img_ptr_t *out, int width, int height);
int main(int argc, char **argv);

int main(int argc, char **argv)
{
    int width, height, channels;
    image_ptr_t data = stbi_load(argv[1], &width, &height, &channels, 1);
    img_ptr_t input = convert2float(data, width, height);
    stbi_image_free(data);
    img_ptr_t lowest_descent = NULL;
    lowest_descent_kernel(input, &lowest_descent, width, height);
    image_ptr_t output = convert2image(lowest_descent, width, height);
    stbi_write_png("result.png", width, height, channels, output, width * channels);
    free(output);
    free(lowest_descent);
    return 0;
}

img_ptr_t convert2float(image_ptr_t image, int width, int height)
{
    img_ptr_t temp = (img_ptr_t)calloc(width * height, sizeof(img_t));
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            temp[i * width + j] = (img_t)image[i * width + j];
        }
    }
    return temp;
}
image_ptr_t convert2image(img_ptr_t image, int width, int height)
{
    image_ptr_t temp = (image_ptr_t)calloc(width * height, sizeof(image_t));
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            temp[i * width + j] = (image_t)image[i * width + j];
        }
    }
    return temp;
}

void lowest_descent_kernel(img_ptr_t in, img_ptr_t *out, int width, int height)
{
    img_ptr_t _lowest = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (_lowest == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < width - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            // find minimum in neighbors
            img_t min = INFINITY;
            if (min > in[i * width + (j + 1)])
                min = in[i * width + (j + 1)];
            if (min > in[i * width + (j - 1)])
                min = in[i * width + (j - 1)];
            if (min > in[(i + 1) * width + j])
                min = in[(i + 1) * width + j];
            if (min > in[(i - 1) * width + j])
                min = in[(i - 1) * width + j];
            if (min > in[(i - 1) * width + (j + 1)])
                min = in[(i - 1) * width + (j + 1)];
            if (min > in[(i - 1) * width + (j - 1)])
                min = in[(i - 1) * width + (j - 1)];
            if (min > in[(i + 1) * width + (j + 1)])
                min = in[(i + 1) * width + (j + 1)];
            if (min > in[(i + 1) * width + (j - 1)])
                min = in[(i + 1) * width + (j - 1)];
            // check if we have plateaued
            bool exists_q = false;
            img_t p = in[i * width + j];
            if (p > in[i * width + (j + 1)] && in[i * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -(i * width + (j + 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[i * width + (j - 1)] && in[i * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = -(i * width + (j - 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + j] && in[(i + 1) * width + j] == min)
            {
                _lowest[i * width + j] = -((i - 1) * width + j);
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + j] && in[(i - 1) * width + j] == min)
            {
                _lowest[i * width + j] = -((i - 1) * width + j);
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + (j + 1)] && in[(i - 1) * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -((i - 1) * width + (j + 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + (j - 1)] && in[(i - 1) * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = -((i - 1) * width + (j - 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + (j + 1)] && in[(i + 1) * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -((i + 1) * width + (j + 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + (j - 1)] && in[(i + 1) * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = -((i + 1) * width + (j - 1));
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
        FOUND_LOWEST_DESCENT:
            if (exists_q == false)
            {
                _lowest[i * width + j] = PLATEAU;
            }
        }
    }
    *out = _lowest;
}
