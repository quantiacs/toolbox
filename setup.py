from setuptools import setup

setup(
    name="toolbox",
    version="0.0.179",
    url="https://quantiacs.io",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=['xarray', 'pandas', 'numpy', 'scipy', 'tabulate', 'bottleneck', 'numba']
)