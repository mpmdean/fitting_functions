from setuptools import setup

setup(
   name='fitting_functions',
   description='Fitting functions for lmfit',
   author='Mark Dean',
   author_email='mdean@bnl.gov',
   packages=['fitting_functions'],  #same as name
   install_requires=['numpy', 'lmfit'], #external packages as dependencies
)
