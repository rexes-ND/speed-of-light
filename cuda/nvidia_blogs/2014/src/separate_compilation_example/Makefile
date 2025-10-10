objects = main.o particle.o v3.o

all: $(objects)
	nvcc -arch=sm_89 $(objects) -o app

%.o: %.cu
	nvcc -arch=sm_89 -dc $< -o $@

clean:
	rm -f *.o app
