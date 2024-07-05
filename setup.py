from setuptools import setup

setup(
    name="qnt",
    version="0.0.401",
    url="https://quantiacs.com",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data', 'qnt.examples'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=[
        'scipy>=1.14.0',
        'pandas==2.2.2',
        'xarray==2024.6.0',
        'numpy<2.0.0',
        'tabulate>=0.9.0',
        'bottleneck>=1.3.7',
        'numba==0.60.0',
        'progressbar2>=3.55,<4',
        'cftime==1.6.4',
        'plotly==5.22.0',
        'matplotlib==3.9.0'
    ]
)