# README

## Installation Instructions

You can use the Quantiacs library (QNT) for developing locally trading strategies on your computer.

You can follow these easy steps and create an isolated environment on your machine using conda for managing dependencies and avoiding conflicts:

1 . Install anaconda (v2020.02 is recommended): https://www.anaconda.com/products/individual or https://repo.anaconda.com/archive/.


2. Create an isolated environment for developing strategies and install the QNT library together with needed dependencies:
```bash
conda create -n qntdev quantiacs-source::qnt conda-forge::ta-lib conda-forge::dash=1.18 python=3.7
```

Then set your API key. You can find it in your profile on https://quantiacs.com .
```bash
conda env config vars set -n qntdev API_KEY={your_api_key_here}
```

3 . Activate your environment:
```bash
conda activate qntdev
```
For leaving the environment:
```bash
conda deactivate
```
Each time you want to use the QNT library, reactivate the environment.

4 . Develop your strategy. You can use this code as a starting point:

*strategy.py:*
```python
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk

import xarray as xr


def load_data(period):
    data = qndata.futures_load_data(tail=period)
    return data


def strategy(data):
    close = data.sel(field='close')
    sma200 = qnta.sma(close, 200).isel(time=-1)
    sma20 = qnta.sma(close, 20).isel(time=-1)
    return xr.where(sma200 < sma20, 1, -1)


qnbk.backtest(
    competition_type="futures",
    load_data=load_data,
    lookback_period=365,
    test_period=2*365,
    strategy=strategy
)
```

5 . Use this command to start your strategy (replace python with python3 if your default python version is 2):
```bash
python strategy.py
```

If you experience issues with the API key, prepend the following to your python file:
```bash
import os
os.environ['API_KEY'] = "{your_api_key_here}"
```

6 . When you finish with your strategy, you need to upload 
your code the jupyter notebook on https://quantiacs.com .

You can copy and past your code in a jupyter notebook called strategy.ipynb. 
Or you can upload your python file (strategy.py) and run it from a jupyter notebook called strategy.ipynb:
```python
import strategy
```

7 . Run all cells to test your strategy in the jupyter notebbok.

https://quantiacs.com/personalpage/strategies

Fix the errors if it is necessary.


8 . Send your strategy to the Contest from the "Development" page.

https://quantiacs.com/personalpage/strategies

9 . Wait for your strategy pass filters and win the Contest.

