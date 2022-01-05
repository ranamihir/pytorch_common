from setuptools import find_packages, setup

setup(
    # Application info
    name="pytorch_common",
    version="1.5.3",
    author="Mihir Rana",
    author_email="ranamihir@gmail.com",
    description="Repo for common PyTorch code",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    test_suite="tests",
    # Packages that this package requires
    install_requires=[
        "numpy>=1.21.2",
        "pandas>=1.3.4",
        "matplotlib>=3.4.3",
        "dask[dataframe]>=2021.11.1",
        "toolz==0.10.0",
        "scikit-learn>=1.0.1",
        "dill>=0.3.4",
        "munch>=2.5.0",
        "locket==0.2.0",
    ],
    # Optional dependencies
    extras_require={"nlp": ["transformers>=4.15.0"]},  # for NLP related projects
    # Add config and sql files to the package
    # https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    include_package_data=True,
)
