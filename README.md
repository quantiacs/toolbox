# README

## Introduction

Welcome to the Quantiacs Python Trading Library (QNT), a comprehensive platform for quantitative finance and algorithmic
trading. This library is designed for both beginners and seasoned traders, enabling the development and testing of
trading algorithms.

Discover more about Quantiacs:

- **Website**: [Quantiacs.com](https://quantiacs.com)
- **Detailed Documentation**: [Quantiacs Documentation](https://quantiacs.com/documentation/en/)

## Features

- **Backtesting Engine**: Test your strategies with our advanced backtesting engine.
- **Market Data Access**: Access a wide range of financial data, including stocks, futures, and cryptocurrencies.
- **Strategy Optimization**: Enhance the performance of your algorithms.
- **Community and Support**: Join a thriving community of quantitative traders.

## About the Quantiacs Contests

Quantiacs hosts a variety of quant competitions, catering to different asset classes and investment styles:

- [The Classic Quantiacs Futures Contest](https://quantiacs.com/leaderboard/15): A mainstay contest focusing on futures
  trading.
- [The Crypto Bitcoin Futures Contest](https://quantiacs.com/leaderboard/15): Tailored for trading Bitcoin futures.
- [The Crypto Top-10 Long-Only Contest](https://quantiacs.com/leaderboard/16): Concentrating on a long-only strategy in
  the top 10 cryptocurrencies.
- [The Crypto Top-10 Long-Short Contest](https://quantiacs.com/leaderboard/17): Involves both long and short positions
  in the top 10 cryptocurrencies.

### Current Opportunity

- [The Q21 NASDAQ-100 Long-Short Fundamental Contest](https://quantiacs.com/contest): A specialized contest focusing on
  long-short strategies in the NASDAQ-100, emphasizing fundamental analysis.

Since 2014, Quantiacs has hosted numerous quantitative trading contests, allocating over 38 million USD to winning
algorithms in futures markets. Since 2021, the platform has expanded to include contests for predicting futures,
cryptocurrencies, and stocks.

## Using QNT from Github

The Quantiacs library (QNT) is optimized for local strategy development. We recommend using **Conda** for its stability
and ease of managing dependencies.

1. **Install Anaconda**: Download and install Anaconda
   from [Anaconda's official site](https://www.anaconda.com/products/individual).

2. **Create a QNT Development Environment**:
    - Open your terminal and run:
      ```bash
      conda create -n qntdev 'python>=3.10,<3.11' conda-forge::ta-lib
      conda activate qntdev
      pip install 'ipywidgets==7.5' 'plotly==4.14' 'matplotlib==3.8.1' 'dash==1.21.0' git+https://github.com/quantiacs/toolbox.git
      pip install 'cython==0.29.37'
      pip install --no-build-isolation 'pandas==1.2.5'

      ```
    - *Optional*: Prevent auto-activation of this environment:
      ```bash
      conda config --set auto_activate_base false
      ```

3. **API Key Configuration**:
    - Retrieve your API key from your [Quantiacs profile](https://quantiacs.com/personalpage/homepage).

    - Set the API key in your environment:
       ```bash
       conda env config vars set -n qntdev API_KEY={your_api_key_here}
       ```
    - Alternatively, set the API key in your code (useful for IDE compatibility issues):
      ```python
      import os
      os.environ['API_KEY'] = "{your_api_key_here}"
      ```
4. **Using the Environment**:
    - Activate the environment with:
      ```bash
      conda activate qntdev
      ```
    - Deactivate when done using:
      ```bash
      conda deactivate
      ```
    - Always reactivate when returning to development.


5. **Strategy Development**:
    - Develop in your preferred IDE.
    - For Jupyter notebook usage:
      ```bash
      jupyter notebook
      ```


6. **Contest Participation**:
    - Develop and test your strategy, then submit it to Quantiacs contests.

## Using a Conda Environment

> We recommend using **Conda** for its stability and ease of managing dependencies.

Installation instructions are the same as for Using QNT from Github.
In step two, run the command

```bash
conda create -n qntdev quantiacs-source::qnt 'python>=3.10,<3.11' conda-forge::ta-lib
conda activate qntdev
pip install 'ipywidgets==7.5' 'plotly==4.14' 'matplotlib==3.8.1' 'dash==1.21.0'
pip install 'cython==0.29.37'
pip install --no-build-isolation 'pandas==1.2.5'
```

### Updating the conda environment

- Regularly update the QNT library for the latest features and fixes:

```bash
    ## you can remove the old one before that
    # conda remove -n qntdev quantiacs-source::qnt

    conda install -n qntdev quantiacs-source::qnt
```

You can see the library updates [here](https://anaconda.org/quantiacs-source/qnt/files).

## Pip Environment

> Note: While Conda is recommended, Pip can also be used, especially if Conda is not an option.

This one-liner combines the installation of Python, creation of a virtual environment, and installation of necessary
libraries.

### Single Command Setup

1. **One Command Setup**:
    - Ensure you have [`pyenv`](https://github.com/pyenv/pyenv)
      and [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) installed.
    - Run the following command in your terminal:
      ```bash
      pyenv install 3.10.13 && \
      pyenv virtualenv 3.10.13 name_of_environment && \
      pyenv local name_of_environment && \
      python -m pip install 'ipywidgets==7.5' 'plotly==4.14' 'matplotlib==3.8.1' 'dash==1.21.0' git+git://github.com/quantiacs/toolbox.git
      pip install 'cython==0.29.37'
      pip install --no-build-isolation 'pandas==1.2.5'
      ```

   This command will:
    - Install Python 3.10.13.
    - Create a virtual environment named `name_of_environment`.
    - Activate the environment for the current directory.
    - Install the Quantiacs toolbox and other necessary Python libraries.

2. **TA-Lib Installation**:
    - The [TA-Lib library](https://github.com/TA-Lib/ta-lib-python) may need to be installed separately due to its
      specific installation requirements.

3. **Setting the API Key**:
    - Set the Quantiacs API key in your code:
      ```python
      import os
      os.environ['API_KEY'] = "{your_api_key_here}"
      ```
    - Replace `{your_api_key_here}` with the actual API key found in
      your [Quantiacs profile](https://quantiacs.com/personalpage/homepage).

### Updating the pip environment

Use this command in your environment to install the latest version from the git repository:

```
python -m pip install --upgrade git+git://github.com/quantiacs/toolbox.git
```

## Google Colab support

If you want to use Google Colab with a hosted runtime, start with
this [notebook](https://quantiacs.com/documentation/en/_static/colab.ipynb).

This notebook contains the necessary commands to configure a hosted runtime.

If you use colab with a local runtime, then you can use regular conda environment. Go to the head of this page and
follow the instructions for conda.

## Working example Jupyter Notebook or Jupyter Lab

You can open any strategy on [Quantiacs](https://quantiacs.com) and check dependencies.

1. **Launch Jupyter**: Start Jupyter Notebook or Jupyter Lab.

2. **Open a Strategy File**: Load any `.ipynb` strategy file.

3. **List Dependencies**: Execute `!conda list` in a notebook cell to display installed packages.

4. **Review Dependencies**: Ensure all required dependencies for your strategy are present and up-to-date.

This quick check aids in maintaining a robust environment for your Quantiacs trading strategies.

## How to Check the QNT Library

Using the Quantiacs (QNT) library involves a few basic steps to create and run a trading strategy. Here's a guide on how
to do it:

### Step 1: Create a Strategy

The first step in utilizing the qnt library is to define your trading strategy. An example of a simple strategy is
provided below in the `strategy.py` file. This example demonstrates a basic long-short trading strategy based on the
crossing of two simple moving averages (SMAs) with lookback periods of 20 and 200 trading days.

### Example Strategy: Simple Moving Average Crossover

The following Python code illustrates a straightforward implementation of a trading strategy using the qnt library:

```python
# import os
# 
# os.environ['API_KEY'] = "{your_api_key_here}"

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
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False
)

```

### Step 2: Run the Strategy Using the Command

After creating your strategy, the next step is to run it using the Python command line. To execute your strategy, you
can use the following command in your Python environment:

```bash
python strategy.py
```

This command runs the strategy.py script, which contains the defined trading strategy and invokes the backtest function
from the qnt library. It's important to ensure that the Python environment where you run this command has the qnt
library installed and is properly set up to access market data.

**Executing and submitting your strategy:**

1. When you finish with developing your strategy, you need to upload your code in the **Jupyter Notebook environment on
   the Quantiacs webpage.** There are 2 options:

   a) Copy and paste your code inside the cell of a [Jupyter Notebook](https://quantiacs.com/personalpage/strategies):

   b) Upload your python file (for example, **strategy.py**) in your Jupyter environment root directory and type in
   **strategy.ipynb**:

        import strategy

   > Place the installation commands for external dependencies to init.ipynb.

2. Run all cells to test your strategy in the Jupyter Notebook. Fix the errors if it is necessary. It is a good idea to
   run the file **precheck.ipynb**.

3. Send your strategy to the Contest from the [Development](https://quantiacs.com/personalpage/strategies) area on your
   home page by clicking on the **Submit**
   button:
