import logging
import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# read requirements
install_requires=[]
with open("requirements.txt", "r") as f:
    reqs = f.readlines()
    for i in range(len(reqs)):
        req = reqs[i]
        if i < len(reqs)-1:
            req = req[:-1]
        if req[0] is not '#':
            install_requires.append(req)

# install face_recognition package
setup(
     name='people_detection',
     version='1.0.0',
     author="Christian Pratellesi",
     author_email="christian.pratellesi@tuwien.ac.at",
     description="People detection package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/SUS-SS2022/people-detection",
     packages=find_packages(),
     install_requires=install_requires
)
