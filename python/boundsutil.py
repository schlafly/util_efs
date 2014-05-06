from lsd import bhpix, bounds

def inbounds(bounds, lon, lat):
    """
    Returns a mask marking the longitudes and latitudes in bounds.
    
    lon and lat must be in ra/dec for compatibility with bounds!
    """
    x, y = bhpix.proj_bhealpix(lon, lat)
    return bounds.isInsideV(x, y)
