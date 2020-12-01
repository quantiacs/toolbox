import qnt.data as qndata
import qnt.stats as qnstats
import qnt.xr_talib as qnxrtalib
import qnt.forward_looking as qnfl
import xarray as xr

# this function will be called twice
# - with the entire data
# - with the data excluding last year
def strategy():
    data = qndata.load_cryptocurrency_data()

    wma = qnxrtalib.WMA(data.sel(field='close')[::-1], 290)
    sroc = qnxrtalib.ROCP(wma, 35)

    #is_liquid = data.sel(field="is_liquid")
    weights = xr.ones_like(data.sel(field="open")).where(sroc > 0.0125)

    weights = weights / weights.sum("asset", skipna=True)
    return weights.fillna(0.0)

# this function calculte 2 passes and compare overlapping outputs
output = qnfl.load_data_calc_output_and_check_forward_looking(strategy)

data = qndata.load_cryptocurrency_data()
stat = qnstats.calc_stat(data, output, max_periods=252 * 3)
print(stat.to_pandas())
