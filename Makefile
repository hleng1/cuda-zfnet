all:
	nvcc -o test zf_host.cu
clean:
	rm test
