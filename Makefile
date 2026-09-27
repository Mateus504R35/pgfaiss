EXTENSION = faiss_pg
MODULE_big = faiss_pg
OBJS = faiss_pg.o

DATA = faiss_pg--0.2.sql

PG_CONFIG ?= pg_config

# Disable LLVM bitcode generation. This avoids toolchain issues involving
# OpenMP headers when PostgreSQL was built with LLVM support.
NO_BC = 1

CC = g++
CXX = g++

# Faiss is built from the git submodule and installed system-wide by default.
# These variables can be overridden when using another install prefix:
#   make FAISS_PREFIX=/opt/faiss
FAISS_PREFIX ?= /usr/local
FAISS_INCLUDEDIR ?= $(FAISS_PREFIX)/include
FAISS_LIBDIR ?= $(FAISS_PREFIX)/lib

PG_CPPFLAGS += -I$(FAISS_INCLUDEDIR)
CXXFLAGS += -std=c++17 -fPIC

# The rpath keeps the PostgreSQL backend able to locate libfaiss.so when Faiss
# is installed outside the system's default dynamic-linker search paths.
SHLIB_LINK += -L$(FAISS_LIBDIR) -Wl,-rpath,$(FAISS_LIBDIR) -lfaiss -lstdc++ -fopenmp

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
