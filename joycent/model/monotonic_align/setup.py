""" from https://github.com/jaywalnut310/glow-tts """

from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension
import numpy

setup(
    name="monotonic_align",
    ext_modules=cythonize([Extension("core", ["core.pyx"])]),
    include_dirs=[numpy.get_include()]
)
