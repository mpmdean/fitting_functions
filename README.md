# fitting.git
Functions useful for fitting along with examples

Work with this by installing [docker](https://www.docker.com/) and pip and then running

~~~
pip install jupyter-repo2docker
jupyter-repo2docker --editable .
~~~

Change `tree` to `lab` in the URL for JupyterLab.

Reinstall a changed version of fitting_functions into container by executing
`!python setup.py install` within jupyter after running `cd ~` to move into
the home folder. The old package can be removed via
`rm -r /opt/conda/lib/python3.8/site-packages/fitting_functions*`
if needed.