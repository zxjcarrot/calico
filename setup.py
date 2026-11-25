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
all_sources = ['bindings.cpp', 'calico.cpp', 'hnsw.cpp', 'ivf.cpp', 'pdx.cpp', 'diskann.cpp', 'distance_diskann.cpp'] + tpcc_sources
#all_sources = ['bindings.cpp', 'calico.cpp', 'hnsw.cpp'] + tpcc_sources
# --- MODIFICATION START: Added OpenMP flags ---

# We'll handle different compilers/platforms for better compatibility
openmp_compile_args = []
openmp_link_args = []

# For GCC and Clang on Linux/macOS
if sys.platform != "win32":
    openmp_compile_args.append('-fopenmp')
    openmp_link_args.append('-fopenmp')
    #openmp_link_args.append('-lomp')
# For MSVC on Windows
else:
    openmp_compile_args.append('/openmp')
    # MSVC doesn't require a separate link flag for OpenMP

# --- MODIFICATION END ---


# Compiler arguments translated from your Makefile
opt_level = os.getenv("CALICO_OPT_LEVEL", "3")
cpp_args = [
    '-std=c++20',
    f'-O{opt_level}',
    '-DNDEBUG',
    '-g3', # Generates debug info
    '-fnon-call-exceptions',
    '-fasynchronous-unwind-tables',
    '-march=native', # Recommended for performance
    '-fPIC', # Required for creating a shared library (.so file)
    '-ffast-math', # fast math
    '-fno-omit-frame-pointer'
] + openmp_compile_args # Add the OpenMP compile args

# Optional: extra compiler nudges for auto-vectorization.
if sys.platform != "win32" and os.getenv("CALICO_ENABLE_AUTOVEC", "1") == "1":
    cpp_args.extend([
        '-ftree-vectorize',
        '-fopenmp-simd',
        '-fno-math-errno'
    ])

# Optional: control HNSW_DISABLE_OPTIMISTIC_READ via env for ablation builds
if os.getenv("CALICO_HNSW_DISABLE_OPTIMISTIC_READ", "0") == "1":
    cpp_args.append("-DHNSW_DISABLE_OPTIMISTIC_READ")

# Compile the path-cache translation specialization into the hot path.
if os.getenv("PATH_CACHE", "0") == "1":
    cpp_args.append("-DCALICO_PATH_CACHE")

# Bind one validated HashThenArray component for the complete HNSW search.
if os.getenv("CALICO_BOUND_PATH_CACHE", "0") == "1":
    cpp_args.append("-DCALICO_BOUND_PATH_CACHE")

# Specialize direct-array builds so getPageState and HNSW page-state resolution
# contain no runtime mode test.
if os.getenv("TRADHASH") == "0":
    cpp_args.append("-DCALICO_DIRECT_PAGE_STATE")

# Optional: control specialization via env for ablation builds
specialization = os.getenv("CALICO_SPECIALIZATION", "")
if specialization == "calico":
    cpp_args.append("-DCALICO_SPECIALIZATION_CALICO")
elif specialization == "hash":
    cpp_args.append("-DCALICO_SPECIALIZATION_HASH")
# If not set or other value, no specialization macro is defined (uses residentPtr for generality)

page_size = os.getenv("CALICO_PAGE_SIZE", "8192")
cpp_args.append(f"-DCALICO_PAGE_SIZE={page_size}")

# Linker arguments, if any are needed
linker_args = [] + openmp_link_args # Add the OpenMP link args


# Define the Python extension module.
calico_module = Extension(
    'calico_py',  # This is the name of the Python module (import calico_py)
    
    sources=all_sources,

    depends=['calico.hpp', 'hash_tables.hpp', 'third_party/robin_hood.h', 'third_party/concurrentqueue.hpp'],
    
    # Include directories for header files.
    include_dirs=[
        pybind11_include,
        '.',      # Current directory for calico.hpp
        'tpcc'    # Directory for tpcc headers
    ],
    
    # Libraries to link against, translated from -laio, -luring, -lpthread and abseil
    libraries=['aio', 'uring', 'pthread', 'absl_hash', 'absl_raw_hash_set', 'absl_raw_logging_internal'], 
    
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
