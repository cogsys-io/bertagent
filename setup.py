#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages  # , find_namespace_packages
import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
    "build>=0.10.0",
]

test_requirements = [
    "black==23.3.0",
    "codecov==2.1.13",
    "flake8==6.0.0",
    "flake8-debugger==4.1.2",
    "pytest==5.4.3",
    "pytest-cov==4.0.0",
    "pytest-raises==0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "versioneer>=0.28",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "sphinx>=5.3.0",
    "sphinx_rtd_theme>=1.2.0",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "GitPython>=3.1.31",
    "blessed>=1.20.0",
    "srsly>=2.4.0",
    "pandas>=1.4.0",
    "pytz>=2022.1",
    "tzlocal>=4.2",
    "humanfriendly>=10.0",
    "contexttimer>=0.3.3",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="cogsys.io",
    author_email="bertagent@cogsys.io",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Quantify linguistic agency in textual data.",
    entry_points={
        "console_scripts": [
            "bertagent=bertagent.cli.bertagent:main",
        ],
    },
    install_requires=requirements,
    setup_requires=setup_requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="bertagent",
    name="bertagent",
    packages=find_packages(
        include=["bertagent", "bertagent.*"],
        exclude=["tests", "*.tests", "*.tests.*"],
    ),
    test_suite="bertagent.tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/cogsys-io/bertagent",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
