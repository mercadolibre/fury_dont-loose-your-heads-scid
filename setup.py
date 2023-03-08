from setuptools import setup

from core.install_packages import install_system_packages

VERSION = "0.0.1dev1"

install_system_packages()

TEST_REQUIREMENTS = [
    "codecov==2.0.15",
    "coverage==4.5.2",
    "pytest==4.0.0",
    "pytest-cov==2.6.0",
    "pytest-mock==1.10.4",
]

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
    tests_require=TEST_REQUIREMENTS,
)
