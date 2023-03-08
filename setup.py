from setuptools import setup

VERSION = "0.0.1dev1"

setup(
    name="scid",
    version=VERSION,
    description="Session Conditioned Item Descriptor",
    author="Pablo Zivic, Jorge Sanchez, Rafael Carrascosa",
    author_email="",
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    packages=["scid"],
    install_requires=[
        # pytorch_version
    ],
)
