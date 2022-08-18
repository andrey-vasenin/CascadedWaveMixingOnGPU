CC=clang++
DEBUG_CFLAGS=-Wall -g -std=c++2a -Og -o solver
RELEASE_CFLAGS=-Wall -shared -fPIC -std=c++2a -O3 -Ofast -o $(EXECUTABLE)
CFLAGS=$(DEBUG_CFLAGS)
LDFLAGS=
EXECUTABLE=solver$(shell python3-config --extension-suffix)
CBLAS="/opt/slate/include"
# PYBIND11="/home/labiks/.local/lib/python3.8/site-packages/pybind11/include"
# PYTHON="/usr/include/python3.8"
PYBIND11 = 
PYTHON=""
ROCM="/opt/rocm-5.2.0/include"
HEADERS_INCLUDES = -I$(CBLAS) $(shell python3 -m pybind11 --includes) -I$(ROCM)
ROCM_LIB="/opt/rocm-5.2.0/lib"
LIB_INCLUDES = -L $(ROCM_LIB)

all: main.cpp
	$(CC) $(HEADERS_INCLUDES) $(LIB_INCLUDES) $(RELEASE_CFLAGS) main.cpp -lblas -lfftw3f -lrocblas -lamdhip64 -lhsa-runtime64 -ltbb -lrocsolver

debug: deb.cpp
	$(CC) $(HEADERS_INCLUDES) $(LIB_INCLUDES) $(DEBUG_CFLAGS) deb.cpp -lblas -lfftw3f -lrocblas -lamdhip64 -lhsa-runtime64 -ltbb -lrocsolver

clean:
	rm *.o
	rm $(EXECUTABLE)