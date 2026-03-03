from setuptools import setup, find_namespace_packages

setup(
    name="iqsi",
    version="0.0.1",
    packages=["iqsi"],
    python_requires=">=3.12",
    install_requires=[
        "ml_collections",
        "accelerate",
    ]
)