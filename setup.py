from setuptools import setup

setup(
    name="qnt",
    version="0.0.507",
    url="https://quantiacs.com",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data', 'qnt.examples'],
    package_data={'qnt': ['*.ipynb']},
    install_requires=[
        'scipy>=1.14.0,<2.0.0',
        'pandas>=2.2,<2.3',
        'xarray>=2024.11.0,<2026.0.0',
        'numpy>=1.26.0,<2.3.0',
        'tabulate>=0.9.0,<1.0.0',
        'bottleneck>=1.4.2,<2.0.0',
        'numba>=0.59.0,<1.0.0',
        'progressbar2>=3.55,<5',
        'cftime>=1.6.4,<2.0.0',
        'plotly>=5.22,<7.0',
        'matplotlib>=3.10.0,<4.0.0'
    ]
)
