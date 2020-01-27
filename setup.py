from setuptools import setup, find_packages

setup(
    # Application info
    name="pytorch_common",

    description="DSP Pipeline",

    # Version number:
    version="1.0",

    # Packages
    packages=find_packages(),

    # Tests (python -m unittest)
    test_suite="tests",

    # packages that this package requires
    install_requires=[
    ],

    # Add config and sql files to the package
    # https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    include_package_data=True,
)
