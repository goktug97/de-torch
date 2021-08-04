#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='detorch',
      version=f'1.0.4',
      description='Minimal PyTorch Library for Differential Evolution',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/goktug97/de-torch',
      packages = ['detorch'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=[
          'numpy',
          'torch',
          'gym',
          'pipcs',
          'mpi4py'],
      python_requires='>=3.7',
      include_package_data=True)
