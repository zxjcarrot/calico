import sys
from setuptools import setup, Extension
import pybind11
import glob # Used to find all source files in a directory

# Find all C++ source files in the tpcc directory
tpcc_sources = glob.glob('tpcc/*.pp')

# Combine all C++ source files needed for the module.
# You need bindings.cpp (for Python interface) and all your original source files.
all_sources = ['bindings.cpp', 'calico.cpp', 'hnsw.cpp'] + tpcc_sources

# Compiler arguments translated from your Makefile
cpp_args = [
    '-std=c++20',
    '-O3',
    '-DNDEBUG',
    '-g', # Generates debug info
    '-fnon-call-exceptions',
    '-fasynchronous-unwind-tables',
    '-march=native', # Recommended for performance
    '-fPIC' # Required for creating a shared library (.so file)
]

# Linker arguments, if any are needed (most things go into libraries)
linker_args = []


# Define the Python extension module.
calico_module = Extension(
    'calico_py',  # This is the name of the Python module (import calico_py)
    
    sources=all_sources,
    
    # Include directories for header files.
    include_dirs=[
        pybind11.get_include(),
        '.',      # Current directory for calico.hpp
        'tpcc'    # Directory for tpcc headers
    ],
    
    # Libraries to link against, translated from -laio and -lpthread
    libraries=['aio', 'pthread'], 
    
    language='c++',
    
    # Pass the compiler and linker arguments
    extra_compile_args=cpp_args,
    extra_link_args=linker_args,
)

setup(
    name='calico_py',
    version='0.1',
    description='Python bindings for Calico Index HNSW',
    ext_modules=[calico_module],
)