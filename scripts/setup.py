from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(["./src/core/prox_solver.pyx", "./src/core/util.pyx"], annotate=True, language='c++', gdb_debug=True),
    include_dirs=[np.get_include()]
)

setup(
    ext_modules = cythonize("./src/core/prox_solver_back.pyx", annotate=True, language='c++', gdb_debug=True),
    include_dirs=[np.get_include()]
)

setup(
    ext_modules = cythonize("./src/core/util.pyx", annotate=True, language='c++', gdb_debug=True),
    include_dirs=[np.get_include()]
)

# usage: python -m scripts.setup build_ext --inplace
