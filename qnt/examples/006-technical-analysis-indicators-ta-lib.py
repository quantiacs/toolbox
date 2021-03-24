# This is an adapter for ta-lib
# It allows using xarrays.
# See: qnt.xr_talib
# There are a few examples how to use it, there are much more functions in  qnt.xr_talib

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import qnt.data as qndata
import qnt.xr_talib as xrta

data = qndata.futures.load_data(assets=['F_GC', 'F_DX'], tail=120)
high = data.sel(field='close')
low = data.sel(field='low')
close = data.sel(field='close')
volume = data.sel(field='vol')

stoch = xrta.STOCHF(data, 5, 3, xrta.MaType.EMA)
print("STOCHASTIC")
print("k")
print(stoch.sel(field='fastk').to_pandas().tail().T)
print("d")
print(stoch.sel(field='fastd').to_pandas().tail().T)

ema = xrta.EMA(close, 5)
print("EMA")
print(ema.to_pandas().tail().T)

crow2 = xrta.CDL2CROWS(data)  # candle analysis / pattern recognition
print("2 crows")
print(crow2.to_pandas().tail().T)

# there are much more functions in  qnt.xr_talib
# You can use ta-lib directly, but in this case you need to convert xarray to numpy and back.
