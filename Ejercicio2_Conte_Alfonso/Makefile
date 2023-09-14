#Makefile

# CUDA code generation flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_SM35    := 
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   ?= $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SMXX)
OBJ = Ejercicio_op_vectorial
SRC = Ejercicio_op_vectorial.cu

$(OBJ):$(SRC)
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) $(SRC) -o $(OBJ)

clean: 
	rm $(OBJ) *.o *~
