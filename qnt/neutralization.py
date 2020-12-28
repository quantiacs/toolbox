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
        asset_names = [a['id'] for a in assets]
        no_info_assets = [a for a in weights.asset.to_pandas().values if a not in asset_names]

        groups = set(a.get(group) for a in assets)
        groups = dict((g, [a['id'] for a in assets if a.get(group) == g]) for g in groups)
        
        if len(no_info_assets) > 0:
            groups['no_info']  = no_info_assets
            warnings.warn("Some stocks has no specification. Perhaps you are using illiquid instruments or outdated assets data.")
        
        
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
