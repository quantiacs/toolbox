import warnings

def neutralize(weights, assets, group = 'market'):
    """
    :param weights: xarray with weights of the algorithm
    :param assets: qndata.load_assets
    :param group: neutralize positions by 'market', 'industry' or 'sector'
    :return: xarray with neutrlized positions
    """
    result = weights.copy(True)

    if group in ['industry','sector']:

        assets = [a for a in assets if a['id'] in weights.asset]

        if len(assets) < len(weights.asset):
            raise Exception("The assets information mast be up to date. The assets length is less than stocks number in weigths")

        groups = set(a.get(group) for a in assets)
        groups = dict((g, [a['id'] for a in assets if a.get(group) == g]) for g in groups)

        for j in groups.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result.loc[{'asset':groups[j]}] = result.sel(asset = groups[j]) - result.sel(asset = groups[j]).mean('asset')

    elif group == 'market':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = result - result.mean('asset')
    else:
        raise Exception(f"No such group '{group}'. Use 'market', 'sector' or 'industry' instead.")

    return result
