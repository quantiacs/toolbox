from setuptools import setup

setup(
    name="qnt",
    version="0.0.225",
    url="https://quantiacs.com",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=['xarray', 'pandas', 'numpy', 'scipy', 'tabulate', 'bottleneck', 'numba', 'progressbar2']
)