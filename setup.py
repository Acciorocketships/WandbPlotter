from setuptools import setup
from setuptools import find_packages

setup(
    name="wandb_plotter",
    version="0.0.2",
    packages=find_packages(),
    install_requires=["matplotlib==3.5.0", "pandas", "wandb"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A Data Fetching and Plotting Library for wandb",
)
