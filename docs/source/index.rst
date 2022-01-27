.. dparcel documentation master file, created by
   sphinx-quickstart on Wed Dec 15 14:19:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**dparcel**
*************
.. image:: https://github.com/tschanzer/dparcel/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/tschanzer/dparcel/actions/workflows/python-package.yml
.. image:: https://codecov.io/gh/tschanzer/dparcel/branch/main/graph/badge.svg?token=HNHGRDKPT8
    :target: https://codecov.io/gh/tschanzer/dparcel
.. image:: https://readthedocs.org/projects/dparcel/badge/?version=latest
    :target: https://dparcel.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/github/license/tschanzer/dparcel
    :alt: GitHub

|
:code:`dparcel` is a very simple model of downdrafts in atmospheric convection that uses parcel theory to simulate the motion of a descending, entraining air parcel.
You may supply any atmospheric sounding and initial conditions for the calculation.

It was developed by Thomas Schanzer, an undergraduate student at the University of New South Wales, as part of a research project under the supervision
of Prof. Steven Sherwood of the UNSW Climate Change Research Centre.

Installation
--------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`_ to install :code:`dparcel`.

.. code-block:: console

    pip install dparcel

Dependencies
--------------
The following packages must be installed:

* :code:`numpy`
* :code:`scipy`
* :code:`metpy>=1.2`

Documentation Contents
----------------------
.. toctree::
   :maxdepth: 3

   api/api
   examples/examples
   technical_reference/technical_reference

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
---------

`BSD-3-Clause License <https://choosealicense.com/licenses/bsd-3-clause/>`_
