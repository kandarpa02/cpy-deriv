from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os

ext_modules = cythonize([
    Extension(
        name="deriv.optim._internals._csgd.sgd_step",
        sources=["deriv/optim/_internals/_csgd/sgd_step.pyx"],
        language="c++"
    )
])

setup(
    name="deriv",
    version="0.0.1a1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="A highly efficient autodiff library with a NumPy-like API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/cpy-deriv.git",
    packages=find_packages(),
    ext_modules=ext_modules,
    setup_requires=["cython"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache-2.0",
    zip_safe=False,
)
