import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bggcohomology",
    version="1.6.1",
    author="Rik Voorhaar",
    description="Tools for the BGG complex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RikVoorhaar/bgg-cohomology",
    ext_modules=cythonize("bggcohomology/*.pyx"),
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="sagemath mathematics flag-varieties representation-theory algebra",
    extras_require={
        "passagemath": [
            "passagemath-graphs",
            "passagemath-groups",
            "passagemath-modules",
            "passagemath-repl",
        ],
    },
)
