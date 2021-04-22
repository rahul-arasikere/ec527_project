#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv)
{
    int x, y, n;
    unsigned char *data = stbi_load(argv[1], &x, &y, &n, 0);
    stbi_write_bmp("result.bmp", x, y, n, data);
    return 0;
}