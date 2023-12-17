from setuptools import setup
from Cython.Build import cythonize

setup(
    name='PyRPS',
    ext_modules=cythonize('simulation.pyx')
)