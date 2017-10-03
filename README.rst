*********
quickspec
*********

- Version: 0.1
- *Author:* Daniel Lenz, Duncan Hanson

Code to estimate the angular power spectra of cosmic fields under the limber approximation.

This is originally based on the `quickspec <https://github.com/dhanson/quickspec>`_ package from Duncan Hanson. Our goal is to extend the legacy value of this package by adding tests, continuous integration, wide-ranging compatibility, and documentation.


Project Status
==============

.. image:: https://travis-ci.org/DanielLenz/quickspec.svg?branch=master
    :target: https://travis-ci.org/DanielLenz/quickspec
    :alt: Quickspec's Travis CI Status

.. image:: https://coveralls.io/repos/github/DanielLenz/quickspec/badge.svg?branch=master
    :target: https://coveralls.io/github/DanielLenz/quickspec?branch=master
    :alt: Quickspec's Coveralls Status


.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

`quickspec` is still in the early-development stage. While much of the
functionality is already working as intended, the API is not yet stable.
Nevertheless, we kindly invite you to use and test the library and we are
grateful for feedback. Note, that work on the documentation is still ongoing.

Usage
=====

Examples and Documentation
--------------------------

We are currently still working on the docs and they are not available yet. However, you can find some examples on how to use `quickspec` in these `notebooks <http://nbviewer.jupyter.org/github/daniellenz/quickspec/blob/master/notebooks/index.ipynb>`_.

Testing
-------

After installation (see below) you can test, if everything works as intended::

  import quickspec

  quickspec.test()

Alternatively, you can clone the repository and run::

  python setup.py test

License
=======

This project is Copyright (c) Daniel Lenz and licensed under the terms of the
BSD 3-Clause license. See the licenses folder for more information.

Installation
============

For now, the installation is only possible from source. Download the tar.gz-file,
extract (or clone from GitHub) and simply execute::

    python setup.py install

Dependencies
------------

We kept the dependencies as minimal as possible. Aside from very conservative requirements such as numpy and scipy, `camb <http://camb.readthedocs.io/en/latest/>`_ is required to compute the matter power spectrum. The full list of the required packages is:

* setuptools
* numpy 1.11 or later
* astropy 1.3 or later (2.0 recommended)
* scipy 0.15 or later
* pytest 2.6 or later
* `camb <http://camb.readthedocs.io/en/latest/>`_

All these packages can easily be obtained and maintained with the `anaconda python distribution <https://www.anaconda.com/download/>`_.

Who do I talk to?
=================

If you encounter any problems or have questions, do not hesitate to raise an
issue or make a pull request. Moreover, you can contact the devs directly:

- *mail@daniellenz.org*
