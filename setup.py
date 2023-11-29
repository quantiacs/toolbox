from setuptools import setup

setup(
    name="qnt",
    version="0.0.303",
    url="https://quantiacs.com",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data', 'qnt.examples'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=[
        'scipy>=1.11.3',
        'pandas==1.3.5',
        'xarray==0.20.2',
        'numpy==1.23.5',
        'tabulate>=0.9.0',
        'bottleneck>=1.3.7',
        'numba==0.58.1',
        'progressbar2>=3.55,<4'
    ]
)