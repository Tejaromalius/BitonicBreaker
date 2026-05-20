CXX=g++
CC=mpicc
CXXFLAGS=-O2 -Wall -std=c++14
CFLAGS=-O2 -Wall

all: sorting_serial sorting_parallel reports

sorting_serial: $(wildcard src/serial/*.cpp src/serial/algorithms/*.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

sorting_parallel: $(wildcard src/parallel/*.c src/parallel/algorithms/*.c)
	$(CC) $(CFLAGS) -o $@ $^

reports:
	mkdir -p reports

clean:
	rm -f sorting_serial sorting_parallel
	rm -rf reports

.PHONY: all clean
