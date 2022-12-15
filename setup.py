# -*- coding: utf-8 -*-
"""
Created on Thu 15 December 2022
"""

from setuptools import find_packages, setup

setup(
    name='myimg',
    packages=find_packages(include=['pyimg']),
    version='0.0.1',
    description='Python library for image pre processing for PIV/PTV applications',
    install_requires=['numpy', 'scipy', 'scikit-image','pandas','matplotlib','pyyaml', 'tk', 'Pillow', 'opencv-python'],
    author='Alec Petersen',
    author_email='alecjpetersen@gmail.com',
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)
