from setuptools import setup

setup(
    name='EEGpip',
    version='0.2',
    py_modules=['main'],
    entry_points={'console_scripts': ['EEGpip = main:main']},
)
