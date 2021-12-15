dparcel
*********

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
* :code:`metpy`

Documentation
---------------
Documentation is available at https://dparcel.readthedocs.io/.

A report that discusses the background and theory of the model, existing literature, the methods used (with flowcharts for the main functions in the package) and
some basic results that were obtained from the model is available `here <https://github.com/tschanzer/dparcel/blob/main/docs/report.pdf>`_.

License
---------

`BSD-3-Clause License <https://choosealicense.com/licenses/bsd-3-clause/>`_
