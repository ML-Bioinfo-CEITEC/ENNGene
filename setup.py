#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'deepnet'
DESCRIPTION = 'DNN app for genomic data.'
URL = 'https://gitlab.com/RBP_Bioinformatics/deepnet.git'
KEYWORDS = ['dnn', 'genomics']
EMAIL = 'chalupovaeliska@email.cz'
AUTHOR = 'Eliška Chalupová & RBP Bioinformatics team at CEITEC MU'
REQUIRES_PYTHON = '>=3.6.0,<=3.7.0'
VERSION = '0.1.0'
REQUIRED = [
    'tensorflow-gpu>=2.0.0',
    'matplotlib>=3.1.1',
    'pydot>=1.3.0',
    'pandas>=1.0.0'
    'streamlit>=0.52.2'
    # TODO how to add pckgs not available through pypi?
    # vienarna
    # graphviz
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    keywords=KEYWORDS,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages('deepnet'),
    install_requires=REQUIRED,
    include_package_data=True,
    license='GNU Affero General Public License v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    cmdclass={
        'upload': UploadCommand,
    },
)
