import sys
import os
from setuptools import setup, Extension
import glob # Used to find all source files in a directory

# Get pybind11 include directory
import pybind11
pybind11_include = pybind11.get_include()

# Find all C++ source files in the tpcc directory
tpcc_sources = glob.glob('tpcc/*.cpp') # Fixed the extension from .pp to .cpp, assuming it was a typo.

# Combine all C++ source files needed for the module.
# You need bindings.cpp (for Python interface) and all your original source files.
all_sources = ['bindings.cpp', 'calico.cpp', 'hnsw.cpp', 'diskann.cpp'] + tpcc_sources

# --- MODIFICATION START: Added OpenMP flags ---

# We'll handle different compilers/platforms for better compatibility
openmp_compile_args = []
openmp_link_args = []

# For GCC and Clang on Linux/macOS
if sys.platform != "win32":
    openmp_compile_args.append('-fopenmp')
    openmp_link_args.append('-fopenmp')
# For MSVC on Windows
else:
    openmp_compile_args.append('/openmp')
    # MSVC doesn't require a separate link flag for OpenMP

# --- MODIFICATION END ---


# Compiler arguments translated from your Makefile
cpp_args = [
    '-std=c++20',
    '-O3',
    '-DNDEBUG',
    '-g3', # Generates debug info
    '-fnon-call-exceptions',
    '-fasynchronous-unwind-tables',
    '-march=native', # Recommended for performance
    '-fPIC' # Required for creating a shared library (.so file)
] + openmp_compile_args # Add the OpenMP compile args

# Linker arguments, if any are needed
linker_args = [] + openmp_link_args # Add the OpenMP link args


# Define the Python extension module.
calico_module = Extension(
    'calico_py',  # This is the name of the Python module (import calico_py)
    
    sources=all_sources,
    
    # Include directories for header files.
    include_dirs=[
        pybind11_include,
        '.',      # Current directory for calico.hpp
        'tpcc'    # Directory for tpcc headers
    ],
    
    # Libraries to link against, translated from -laio, -luring and -lpthread
    libraries=['aio', 'uring', 'pthread'], 
    
    language='c++',
    
    # Pass the compiler and linker arguments
    extra_compile_args=cpp_args,
    extra_link_args=linker_args,
)

setup(
    name='calico_py',
    version='0.1',
    description='Python bindings for Calico Index HNSW and DiskANN',
    ext_modules=[calico_module],
)