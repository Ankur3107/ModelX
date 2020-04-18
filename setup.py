#!/usr/bin/env python

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='model_X',  
     version='0.1.4',
     author="Ankur Singh",
     author_email="ankur310794@gmail.com",
     description="This package contains collection of models",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Ankur3107/ModelX",
     packages=['model_X'],
     install_requires=[], 
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent"
     ]
 )