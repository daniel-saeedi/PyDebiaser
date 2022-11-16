from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="PyDebiaser",
    version="1.0",
    description="A python debiasing library for Transformers models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Saeedi & Kunwar M. Saaim",
    author_email="saeedi.danial@gmail.com",
    url="https://github.com/daniel-saeedi/PyDebiaser",
    python_requires='>=3.7, <4',
    packages=find_packages(include=['PyDebiaser', 'PyDebiaser.*']),
    install_requires=[
        'detoxifier >= 0.2',
        "transformers != 4.18.0",
        "torch >= 1.7.0",
        "sentencepiece >= 0.1.94",
    ],
)