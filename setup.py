from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="PyDebiaser",
    version="1.0.0",
    description="A python debiasing library for Transformers models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Saeedi & Kunwar M. Saaim",
    author_email="saeedi.danial@gmail.com",
    url="https://github.com/daniel-saeedi/PyDebiaser",
    python_requires='>=3.7, <4',
    packages=find_packages(include=['pydebiaser', 'pydebiaser.*']),
    install_requires=[
        'detoxify >= 0.2',
        "sentencepiece >= 0.1.94",
        "torch==1.10.2",
        "transformers==4.16.2",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
        "nltk==3.7.0",
        "datasets==1.18.3",
        "accelerate==0.5.1",
        "wget"
    ],
    include_package_data=True
)

import nltk
nltk.download('punkt')