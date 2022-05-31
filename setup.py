from setuptools import setup, find_packages

import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md")) as f:
    README = f.read()

setup(
    name="pysnapping",
    version="0.0",
    description="Snap points to a line string keeping a given order intact",
    long_description=README,
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    install_requires=["shapely", "numpy", "scipy", "pyproj", "cvxpy"],
)
