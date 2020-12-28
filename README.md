QNT library for strategies development.

You can use this library on your PC.
For this purpose, do these steps: 

1 . Install anaconda (v2020.02 is recommended): https://www.anaconda.com/products/individual

2 . Create an isolated environment for strategies development:
```bash
conda create -n qntdev quantiacs-source::qnt conda-forge::ta-lib conda-forge::dash=1.18
```

Then set your API key. You can find it in your profile on https://quantiacs.io .
```bash
conda env config vars set -n qntdev API_KEY={your_api_key_here}
```

3 . Activate your environment:
```bash
conda activate qntdev
```
You should use this command to reactivate the isolated environment.

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

5 . Use this command to start your strategy:
```bash
API_KEY='{your_api_key}' python3 strategy.py
```

6 . When you finish with your strategy, you need to upload 
your code the jupyter notebook on https://quantiacs.io .

You can copy and past your code in jupyter notebook. 
Or you can upload your python file (strategy.py) and run it from jupyter notebook:
```python
import strategy
```

7 . Run all cells to test your strategy in the jupyter notebbok.

https://quantiacs.io/personalpage/strategies

Fix the errors if it is necessary.


8 . Send your strategy to the Contest from the "Development" page.

https://quantiacs.io/personalpage/strategies

9 . Wait for your strategy pass filters and win the Contest.