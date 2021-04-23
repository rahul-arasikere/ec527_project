CC = gcc
CFLAGS = -Wall -g
PROGS = watershed
OBJS = main.o
LD_LIBRARY = -lm

all: $(PROGS)

watershed: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LD_LIBRARY)

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $^ $(LD_LIBRARY)

clean:
	rm -rf $(PROGS) $(OBJS) *.png