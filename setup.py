from setuptools import setup, find_packages
setup(
    name="darwin_likelihood",
    version="1",
    packages=['darwin_likelihood'],
    package_dir={'darwin_likelihood': 'darwin_likelihood'},
    package_data = {"darwin_likelihood":["data/*"]},
    include_package_data=True,
    author='Knut Morå',
    author_email="knut.dundas.mora@columbia.edu",
    description="blueice-based WIMP inference for Darwin",

)
