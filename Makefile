all:
	nvcc -o zfnet zf_host.cu
clean:
	rm zfnet