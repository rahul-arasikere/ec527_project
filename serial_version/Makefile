CC = clang++
CFLAGS = -Wall -g
PROGS = watershed
OBJS = main.o
LD_LIBRARY = -lm

all: $(PROGS)

watershed: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LD_LIBRARY)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $^ $(LD_LIBRARY)

clean:
	rm -rf $(PROGS) $(OBJS) *.png