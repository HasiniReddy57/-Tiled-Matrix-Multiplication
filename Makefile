CC = nvcc
CFLAGS = -arch=sm_86

all: hpatloll_lab1

hpatloll_lab1: hpatloll_lab1.cu
    $(CC) $(CFLAGS) hpatloll_lab1.cu -o hpatloll_lab1

clean:
    rm -f hpatloll_lab1
