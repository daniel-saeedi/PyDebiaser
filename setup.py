from setuptools import setup

setup
(
    name='PyDebiaser',
    version='1.0',
    author='Daniel Saeedi & Kunwar M. Saaim',
    description='A debiasing library for Transformers models',
    long_description='A debiasing library for Transformers models',
    url='https://github.com/daniel-saeedi/PyDebiaser',
    keywords='debiaser, hard-debias, sent-debias, self-debias, top-k',
    python_requires='>=3.7, <4',
    packages=find_packages(include=['PyDebiaser', 'PyDebiaser.*']),
    install_requires=[
        'detoxifier >= 0.2',
        "transformers != 4.18.0",
        "torch >= 1.7.0",
        "sentencepiece >= 0.1.94",
    ],
)