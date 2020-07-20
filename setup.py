import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyGT",
    version="0.1.0",
    author="TD Swinburne",
    author_email="swinburne@cinam.univ-mrs.fr",
    description="Stable analysis of metastable Markov chains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomswinburne/PyGT",
    packages=setuptools.find_packages(),
    install_requires=['scipy>=1.5','numpy>=1.17','matplotlib>=3','tqdm>=4.47','pathos>=0.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
