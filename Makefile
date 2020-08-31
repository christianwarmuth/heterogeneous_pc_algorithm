all: cupc cupc_counter cupc_heterogeneous

cupc: ./src/cupc.cu
		mkdir -p ./build
		nvcc -O3 --shared -Xcompiler -fPIC -o ./build/cupc.so ./src/cupc.cu

cupc_counter: ./src/cupc_counter.cu
		mkdir -p ./build
		nvcc -O3 --shared -Xcompiler -fPIC -o ./build/cupc_counter.so ./src/cupc_counter.cu

cupc_heterogeneous: ./src/cupc_heterogeneous.cu ./src/constraint_heterogeneous.cpp
		mkdir -p ./build
		nvcc -O3 --shared -Xcompiler "-larmadillo -fopenmp -fPIC" -o ./build/cupc_heterogeneous.so ./src/cupc_heterogeneous.cu ./src/constraint_heterogeneous.cpp

clean:
		rm -rf ./build
