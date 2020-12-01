import qnt.data as qndata
import qnt.stats as qnstats
import qnt.xr_talib as qnxrtalib
import qnt.forward_looking as qnfl
import time

data = qndata.load_data(min_date="2010-01-01", max_date=None, forward_order=True, dims=("time", "field", "asset"))


def strategy(data):
    wma = qnxrtalib.WMA(data.sel(field='close'), 290)
    sroc = qnxrtalib.ROCP(wma, 35)

    is_liquid = data.sel(field="is_liquid")
    weights = is_liquid.where(sroc > 0.0125)

    weights = weights / weights.sum("asset", skipna=True)
    return weights.fillna(0.0)


t0 = time.time()
output = qnfl.calc_output_and_check_forward_looking(data, strategy)
t1 = time.time()
print(t1 - t0)
stat = qnstats.calc_stat(data, output, max_periods=252 * 3)
t2 = time.time()
print(t2 - t1)
stat2 = qnstats.calc_stat(data, output, max_periods=252 * 3, per_asset=True)
t3 = time.time()
print(t3 - t2)

print(stat2.sel(field='sharpe_ratio').transpose().to_pandas())

qndata.write_output(output)
