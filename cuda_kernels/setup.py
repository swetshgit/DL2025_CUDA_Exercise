"""
setup.py — builds the CUDA extension as a Python-importable shared library.

Usage:
    cd cuda_kernels/
    python setup.py build_ext --inplace

After building you'll find a file like:
    temporal_features_ext.cp312-win_amd64.pyd   (Windows)
    temporal_features_ext.cpython-312-x86_64-linux-gnu.so  (Linux)

That file IS the Python module — import it directly:
    import temporal_features_ext
"""

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="temporal_features_ext",

    # CUDAExtension tells setuptools to use nvcc for .cu files
    # and the host C++ compiler for .cpp files, then link them together.
    ext_modules=[
        CUDAExtension(
            name="temporal_features_ext",   # the Python import name
            sources=[
                "temporal_features_cpu.cpp",  # pybind11 entry + CPU fallback
                "temporal_features.cu",       # your CUDA kernels
            ],
            extra_compile_args={
                # nvcc flags — passed to the CUDA compiler
                "nvcc": [
                    "-O2",            # optimisation level
                    "--use_fast_math",# allow approximate math (fine for ML)
                    "-std=c++17",     # C++ standard
                    # Uncomment to target your specific GPU (Ampere = 8.6):
                    # "-gencode=arch=compute_86,code=sm_86",
                ],
                # cxx flags — passed to the host C++ compiler (MSVC or GCC)
                "cxx": [
                    "/O2" if __import__("sys").platform == "win32" else "-O2",
                    "/std:c++17" if __import__("sys").platform == "win32" else "-std=c++17",
                ],
            },
        )
    ],

    # Let torch's custom build command handle compilation
    cmdclass={"build_ext": BuildExtension},
)
