#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PLATEAU 0

const int neighbour_x[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
const int neighbour_y[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

typedef unsigned char image_t, *image_ptr_t;
typedef int img_t, *img_ptr_t;

img_ptr_t convert2data(image_ptr_t image, int width, int height);
image_ptr_t convert2image(img_ptr_t image, int width, int height);
void steepest_descent_kernel(img_ptr_t in, img_ptr_t *out, int width, int height);
void border_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height);
void minima_basin_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height);
void watershed_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height);
double interval(struct timespec start, struct timespec end);
int main(int argc, char **argv);

int main(int argc, char **argv)
{
    int width, height, channels;
    image_ptr_t data = stbi_load(argv[1], &width, &height, &channels, 1);
    img_ptr_t input = convert2data(data, width, height);
    stbi_image_free(data);
    img_ptr_t lowest_descent = NULL;
    img_ptr_t border = NULL;
    img_ptr_t minima = NULL;
    struct timespec time_start, time_stop;
    img_ptr_t watershed = NULL;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start); // start timer
    steepest_descent_kernel(input, &lowest_descent, width, height);
    border_kernel(input, lowest_descent, &border, width, height);
    minima_basin_kernel(input, border, &minima, width, height);
    watershed_kernel(input, minima, &watershed, width, height);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    printf("Process took %f seconds\n", interval(time_start, time_stop));
    stbi_write_png("1_lowest_descent_result.png", width, height, channels, convert2image(lowest_descent, width, height), width * channels);
    stbi_write_png("2_border_result.png", width, height, channels, convert2image(border, width, height), width * channels);
    stbi_write_png("3_minima_basin_result.png", width, height, channels, convert2image(minima, width, height), width * channels);
    stbi_write_png("4_watershed_result.png", width, height, channels, convert2image(watershed, width, height), width * channels);
    free(watershed);
    free(lowest_descent);
    free(border);
    free(input);
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

img_ptr_t convert2data(image_ptr_t image, int width, int height)
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

image_ptr_t convert2image(img_ptr_t image, int width, int height)
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

void steepest_descent_kernel(img_ptr_t in, img_ptr_t *out, int width, int height)
{
    img_ptr_t _lowest = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (_lowest == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            // find minimum in neighbors
            img_t min = (img_t)INFINITY;
            for (int k = 0; k < 8; k++)
            {
                if (min > in[(i + neighbour_x[k]) * width + (j + neighbour_y[k])])
                {
                    min = in[(i + neighbour_x[k]) * width + (j + neighbour_y[k])];
                }
            }
            // check if we have plateaued
            bool exists_q = false;
            img_t p = in[i * width + j];
            for (int k = 0; k < 8; k++)
            {
                img_t q = in[(i + neighbour_x[k]) * width + (j + neighbour_y[k])];
                if (p > q && q == min)
                {
                    _lowest[i * width + j] = -q;
                    exists_q = true;
                    break;
                }
            }
            if (!exists_q)
            {
                _lowest[i * width + j] = (img_t)PLATEAU;
            }
        }
    }
    *out = _lowest;
}

void border_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height)
{
    img_ptr_t _border = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (in == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    bool stable = false;
    img_ptr_t temp_border = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (temp_border == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    };
    while (!stable)
    {
        stable = true;
        memcpy(temp_border, _border, width * height * sizeof(img_t));
        for (int i = 1; i < height - 1; i++)
        {
            for (int j = 1; j < width - 1; j++)
            {
                int p = i * width + j;
                if (in[p] == (img_t)PLATEAU)
                {
                    for (int k = 0; k < 8; k++)
                    {
                        int q = ((i + neighbour_x[k]) * width + (j + neighbour_y[k]));
                        if (in[q] < 0 && image[q] == image[p])
                        {
                            if (temp_border[p] != -(q))
                                stable = false;
                            temp_border[p] = -(q);
                            goto _kernel_exit_goto;
                        }
                    }
                }
            }
        _kernel_exit_goto:
            continue;
        }

        memcpy(_border, temp_border, width * height * sizeof(img_t));
    }
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (in[i * width + j] == (img_t)PLATEAU)
            {
                _border[i * width + j] = -(i * width + j);
            }
        }
    }
    *out = _border;
}

void minima_basin_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height)
{
    img_ptr_t _minima = (img_ptr_t)calloc(width * height, sizeof(img_t));
    if (_minima == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    memcpy(_minima, in, height * width * sizeof(img_t));
    bool stable = false;
    while (!stable)
    {
        stable = true;
        for (int i = 1; i < height - 1; i++)
        {
            for (int j = 1; j < width - 1; j++)
            {
                if (_minima[i * width + j] > (img_t)PLATEAU)
                {
                    img_t label = (img_t)INFINITY;
                    if (_minima[i * width + (j + 1)] < label && image[i * width + (j + 1)] == image[i * width + j])
                    {
                        label = _minima[i * width + (j + 1)];
                    }
                    if (_minima[i * width + (j - 1)] < label && image[i * width + (j - 1)] == image[i * width + j])
                    {
                        label = _minima[i * width + (j - 1)];
                    }
                    if (_minima[(i + 1) * width + (j + 1)] < label && image[(i + 1) * width + (j + 1)] == image[i * width + j])
                    {
                        label = _minima[(i + 1) * width + (j + 1)];
                    }
                    if (_minima[(i + 1) * width + (j - 1)] < label && image[(i + 1) * width + (j - 1)] == image[i * width + j])
                    {
                        label = _minima[(i + 1) * width + (j - 1)];
                    }
                    if (_minima[(i - 1) * width + (j + 1)] < label && image[(i - 1) * width + (j + 1)] == image[i * width + j])
                    {
                        label = _minima[(i - 1) * width + (j + 1)];
                    }
                    if (_minima[(i - 1) * width + (j - 1)] < label && image[(i - 1) * width + (j - 1)] == image[i * width + j])
                    {
                        label = _minima[(i - 1) * width + (j - 1)];
                    }
                    if (_minima[(i + 1) * width + j] < label && image[(i + 1) * width + j] == image[i * width + j])
                    {
                        label = _minima[(i + 1) * width + j];
                    }
                    if (_minima[(i - 1) * width + j] < label && image[(i - 1) * width + j] == image[i * width + j])
                    {
                        label = _minima[(i - 1) * width + j];
                    }
                    if (label < _minima[i * width + j])
                    {
                        if (_minima[_minima[i * width + j]] != label)
                        {
                            stable = false;
                        }
                        _minima[_minima[i * width + j]] = label;
                    }
                }
            }
        }
        for (int i = 1; i < height - 1; i++)
        {
            for (int j = 1; j < width - 1; j++)
            {
                if (_minima[i * width + j] > (img_t)PLATEAU)
                {
                    img_t label = _minima[i * width + j];
                    img_t ref = (img_t)INFINITY;
                    while (label != ref)
                    {
                        ref = label;
                        label = _minima[ref];
                    }
                    if (label != ref)
                    {
                        stable = false;
                    }
                    _minima[i * width + j] = label;
                }
            }
        }
    }
    *out = _minima;
}

void watershed_kernel(img_ptr_t image, img_ptr_t in, img_ptr_t *out, int width, int height)
{
    img_ptr_t _watershed = (img_ptr_t)calloc(height * width, sizeof(img_t));
    if (_watershed == NULL)
    {
        perror("Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    memcpy(_watershed, in, height * width * sizeof(img_t));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            _watershed[i * width + j] = abs(_watershed[i * width + j]);
        }
    }
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            img_t label = _watershed[i * width + j];
            if (label != (i * width + j))
            {
                img_t ref = (img_t)INFINITY;
                while (ref != label)
                {
                    ref = label;
                    label = _watershed[ref];
                }
                _watershed[i * width + j] = label;
            }
        }
    }
    *out = _watershed;
}