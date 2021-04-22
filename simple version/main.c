#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PLATEAU 0

typedef unsigned char image_t, *image_ptr_t;

void lowest_descent_kernel(image_ptr_t in, image_ptr_t *out, int width, int height);
int main(int argc, char **argv);

int main(int argc, char **argv)
{
    int width, height, n;
    image_ptr_t data = stbi_load(argv[1], &width, &height, &n, 0);
    image_ptr_t lowest_descent = NULL;
    lowest_descent_kernel(data, &lowest_descent, width, height);
    stbi_write_bmp("result.bmp", width, height, n, lowest_descent);
    return 0;
}

void lowest_descent_kernel(image_ptr_t in, image_ptr_t *out, int width, int height)
{
    image_ptr_t _lowest = (image_ptr_t)calloc(width * height, sizeof(image_t));
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
            int min = in[i * width + j];
            if (min < in[i * width + (j + 1)])
                min = in[i * width + (j + 1)];
            if (min < in[i * width + (j - 1)])
                min = in[i * width + (j - 1)];
            if (min < in[(i + 1) * width + j])
                min = in[(i + 1) * width + j];
            if (min < in[(i - 1) * width + j])
                min = in[(i - 1) * width + j];
            if (min < in[(i - 1) * width + (j + 1)])
                min = in[(i - 1) * width + (j + 1)];
            if (min < in[(i - 1) * width + (j - 1)])
                min = in[(i - 1) * width + (j - 1)];
            if (min < in[(i + 1) * width + (j + 1)])
                min = in[(i + 1) * width + (j + 1)];
            if (min < in[(i + 1) * width + (j - 1)])
                min = in[(i + 1) * width + (j - 1)];

            // check if we have plateaued
            bool exists_q = false;
            int p = in[i * width + j];
            if (p > in[i * width + (j + 1)] && in[i * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -in[i * width + (j + 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[i * width + (j - 1)] && in[i * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = -in[i * width + (j - 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + j] && in[(i + 1) * width + j] == min)
            {
                _lowest[i * width + j] = -in[(i - 1) * width + j];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + j] && in[(i - 1) * width + j] == min)
            {
                _lowest[i * width + j] = -in[(i - 1) * width + j];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + (j + 1)] && in[(i - 1) * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -in[(i - 1) * width + (j + 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i - 1) * width + (j - 1)] && in[(i - 1) * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = -in[(i - 1) * width + (j - 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + (j + 1)] && in[(i + 1) * width + (j + 1)] == min)
            {
                _lowest[i * width + j] = -in[(i + 1) * width + (j + 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
            if (p > in[(i + 1) * width + (j - 1)] && in[(i + 1) * width + (j - 1)] == min)
            {
                _lowest[i * width + j] = in[(i + 1) * width + (j - 1)];
                exists_q = true;
                goto FOUND_LOWEST_DESCENT;
            }
        FOUND_LOWEST_DESCENT:
            if (!exists_q)
            {
                _lowest[i * width + j] = PLATEAU;
            }
        }
    }
    *out = _lowest;
}
