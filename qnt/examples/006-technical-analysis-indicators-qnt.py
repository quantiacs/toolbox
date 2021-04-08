# Major technical indicators implemented in qnt.ta
# See: qnt.ta.__init__.py
# Use the source code as an example how to implement fast indicators.

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import qnt.ta as qnta
import qnt.data as qndata

data_all = qndata.futures.load_data(tail=120)
close_all = data_all.sel(field='close')

data = qndata.futures.load_data(assets=['F_GC', 'F_DX'], tail=120)
high = data.sel(field='close')
low = data.sel(field='low')
close = data.sel(field='close')
volume = data.sel(field='vol')

print("Moving averages:")
sma = qnta.sma(close, 20)
ema = qnta.ema(close, 20)
lwma = qnta.lwma(close, 20)
wma = qnta.wma(close, [3, 2, 1])
wi_ma = qnta.wilder_ma(close, 20)
vwma = qnta.vwma(close, volume, 20)

print("SMA(20)")
print(sma.to_pandas().tail().T)


print("---")
print("Oscillators:")
stoch_k = qnta.stochastic_k(high, low, close, 14)
stoch_fast_k, stoch_fast_d = qnta.stochastic(high, low, close, 14)
stoch_slow_k, stoch_slow_d = qnta.stochastic(high, low, close, 14)
rsi = qnta.rsi(close, 14)
roc = qnta.roc(close, 7)
sroc = qnta.sroc(close, 13, 21)
macd_line, macd_signal_line, macd_hist = qnta.macd(close, 12, 26, 9)
trix = qnta.trix(close, 18)

print("ROC")
print(roc.to_pandas().tail().T)

print("STOCHASTIC")
print("k")
print(stoch_fast_k.to_pandas().tail().T)
print("d")
print(stoch_fast_d.to_pandas().tail().T)

print("MACD")
print("line")
print(macd_line.to_pandas().tail().T)
print("signal-line")
print(macd_signal_line.to_pandas().tail().T)
print("histogram")
print(macd_hist.to_pandas().tail().T)


print("---")
print("Index indicators")
atr = qnta.atr(high, low, close, 14)
plus_di, minus_di, adx, adxr = qnta.dms(high, low, close, 14, 20, 7)
print("Directional Movement System")
print("+DI")
print(plus_di.to_pandas().tail().T)
print("-DI")
print(minus_di.to_pandas().tail().T)
print("ADX")
print(adx.to_pandas().tail().T)
print("ADXR")
print(adxr.to_pandas().tail().T)


print("---")
print("Cumulative")
obv = qnta.obv(close, volume)

chaikin_adl = qnta.chaikin_adl(high, low, close, volume)
chaikin_osc = qnta.chaikin_osc(chaikin_adl, 3, 10)
print("Chaikin ADL")
print(chaikin_adl.to_pandas().tail().T)
print("Chaikin oscillator")
print(chaikin_osc.to_pandas().tail().T)


print("---")
print("Global")
ad_line = qnta.ad_line(close_all)
ad_ratio = qnta.ad_ratio(close_all)
print("AD rartio")
print(ad_ratio.to_pandas().tail())


print("---")
print("Others")
pivot_points = qnta.pivot_points(close, 20, 20)
top_pivot_points = qnta.top_pivot_points(close, 20)
bottom_pivot_points = qnta.bottom_pivot_points(close, 20)

change = qnta.change(close, 1)
shift = qnta.shift(close, 1)

std = qnta.std(close)

variance = qnta.variance(close, 20)
asset1, asset2 = close.sel(asset='F_GC'), close.sel(asset='F_DX')
covariance = qnta.covariance(asset1, asset2, 20)
beta = qnta.beta(asset1, asset2, 20)
correlation = qnta.correlation(asset1, asset2, 20)
print("Correlation")
print(correlation.to_pandas().tail())
