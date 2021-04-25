NVCC := nvcc
CFLAGS := -Wall -g
PROGS = watershed
OBJS = main.o
LD_LIBRARY :=

all: $(PROGS)

watershed: $(OBJS)
	$(NVCC) --compiler-options $(CFLAGS) -o $@ $^ $(LD_LIBRARY)

%.o: %.cu
	$(NVCC) --compiler-options $(CFLAGS) -o $@ -c $^ $(LD_LIBRARY)

clean:
	rm -rf $(PROGS) $(OBJS) *.png