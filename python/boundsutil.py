from lsd import bhpix, bounds as lsdbounds
import numpy

def inbounds(bounds, lon, lat):
    """
    Returns a mask marking the longitudes and latitudes in bounds.
    
    lon and lat must be in ra/dec for compatibility with bounds!
    """
    x, y = bhpix.proj_bhealpix(lon, lat)
    inside = numpy.zeros(len(lon), dtype='bool')
    if not isinstance(bounds, list):
        bounds = lsdbounds.make_canonical(bounds)
    for bd in bounds:
        inside = inside | bd[0].isInsideV(x, y)
    return inside
