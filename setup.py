from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="opt",
    version="0.1.0",
    author="Zhe Zhang",
    author_email="zhezhang@slac.stanford.end",
    description="A package for beam-based online optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'gpy',
        'gpyopt',
        'pygmo',
        'smt',
        'pyDOE',
        'fabric'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

