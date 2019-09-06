#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os
import versioneer

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('sklmer', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'sklearn-lmer'
DESCRIPTION = 'Scikit-learn estimator wrappers for pymer4 wrapped LME4 mixed effects models'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'D. Nielson'
MAINTAINER_EMAIL = 'dylan.nielson@gmail.com'
URL = 'https://github.com/nimh-mbdu/sklearn-lmer'
LICENSE = 'CC0'
DOWNLOAD_URL = 'https://github.com/nimh-mbdu/sklearn-lmer'
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'pandas', 'pymer4']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
