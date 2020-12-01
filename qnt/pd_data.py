from qnt.data import load_data as origin_load_data, write_output as origin_write_output, ds


def load_data(assets, min_date='2007-01-01', max_date=None):
    ''' Loads data in dictionary. Key - field (close, open, low, high), value - dataframe (rows - time, columns - tickers)'''
    data = origin_load_data(assets, min_date, max_date, (ds.FIELD, ds.TIME, ds.ASSET))
    res = {}
    for i in data.coords[ds.FIELD].values:
        res[i] = data.sel(**{ds.FIELD: i}).to_pandas()
    return res


def write_output(output):
    ''' Save user output for processing. output - dataframe (rows - time, columns - tickers)  '''
    output.index.name = ds.TIME
    output = output.to_xarray().to_array(ds.ASSET)
    origin_write_output(output)
