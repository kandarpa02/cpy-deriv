from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = cythonize([
    Extension(
        name="deriv.utils.helpers",
        sources=["deriv/utils/helpers.pyx"],
        language="c++"
    ),
    Extension(
        name="deriv.math.math_cpu",
        sources=["deriv/math/math_cpu.pyx"],
        language="c++"
    ),
    Extension(
        name="deriv.math.matrix_cpu",
        sources=["deriv/math/matrix_cpu.pyx"],
        language="c++"
    ),
    Extension(
        name="deriv.nn.non_linear",
        sources=["deriv/nn/non_linear.pyx"],
        language="c++"
    ),
])

setup(
    name="deriv",
    version="0.1.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="A highly efficient autodiff library with a NumPy-like API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/DeepTranslate.git",
    packages=find_packages(),
    ext_modules=extensions,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
