#!/usr/bin/env python3

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    python_requires=">=3.7",
    description="Snap points to a line string keeping a given order intact",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    name="pysnapping",
    packages=find_packages(include=["pysnapping", "pysnapping.*"]),
    zip_safe=False,
    install_requires=[
        "shapely",
        "numpy",
        "pyproj>=3.2.0",  # for defining custom CRS
        "cvxpy",
    ],
)
