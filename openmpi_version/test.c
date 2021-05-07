
void kernel(int* in, int** out, int width, 
            int height)
{
    int* results = (int*)calloc(width * height, 
                    sizeof(int*));
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++){
            // do something here
        }
    }
    out = results;
}

