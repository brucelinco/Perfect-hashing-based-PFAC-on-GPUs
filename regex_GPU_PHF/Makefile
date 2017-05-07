NVCC30   = /usr/local/cuda/bin/nvcc
INC30    = -I/usr/local/cuda/include
LIB30    = -L/usr/local/cuda/lib64
ARGS13   = -arch=sm_13 -Xcudafe=-w -lrt
ARGS20   = -arch=sm_30 -Xcudafe=-w -lrt
ARGS11   = -arch=sm_11 -Xcudafe=-w -lrt
o        = gphf

SRC      = master_kernel.cu
SRC     += main.cc

all :: cuda20

cuda11 :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS11) -o $(o) $(SRC)
	rm -f *.linkinfo
cuda13 :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS13) -o $(o) $(SRC)
	rm -f *.linkinfo
cuda20 :	
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -o $(o) $(SRC) 
	rm -f *.linkinfo
memuse :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -Xptxas=-v -o $(o) $(SRC)
cubin :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -cubin $(SRC)
clean :
	rm -f *.linkinfo
