# Copyright (c) 2021 Thomas Schanzer.
# Distributed under the terms of the BSD 3-Clause License.
"""Setup file for the dparcel package."""

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='dparcel',
    version='0.1',
    description=('A simple parcel theory model of downdrafts '
                 'in atmospheric convection.'),
    long_description=readme(),
    url='https://github.com/tschanzer/dparcel',
    author='Thomas Schanzer',
    author_email='t.schanzer@student.unsw.edu.au',
    license='BSD 3-Clause License',
    packages=['dparcel'],
    install_requires=[
        'numpy',
        'metpy',
        'scipy',
    ],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    keywords='parcel downdraft atmosphere convection model',
    include_package_data=True
)
