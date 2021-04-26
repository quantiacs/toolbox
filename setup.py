from setuptools import setup

setup(
    name="qnt",
    version="0.0.251",
    url="https://quantiacs.com",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data', 'qnt.examples'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=[
        'scipy>=1.4.1',
        'pandas>=1.0.1',
        'xarray>=0.16.0',
        'numpy>=1.18',
        'tabulate>=0.8.3',
        'bottleneck>=1.3.1',
        'numba==0.53',
        'progressbar2>=3.37,<4'
    ]
)