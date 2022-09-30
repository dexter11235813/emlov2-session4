#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="hydra-timm-pl",
    version="0.0.1",
    description="Submission for EMLOv2 session 2",
    author="",
    author_email="",
    url="https://github.com/dexter11235813/emlov2-session2",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
