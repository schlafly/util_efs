import os
import random
import pdb
import fnmatch
import pickle
import numpy
import matplotlib
from matplotlib import pyplot
# from astrometry.libkd.spherematch import match_radec
from util_efs_c import max_bygroup, add_arr_at_ind


# stolen from Mario Juric
def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = numpy.radians(lon1); lat1 = numpy.radians(lat1)
    lon2 = numpy.radians(lon2); lat2 = numpy.radians(lat2)

    return numpy.degrees(
        2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + 
                       cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));


def sample(obj, n):
    ind = random.sample(range(len(obj)),numpy.int(n))
    return obj[numpy.array(ind)]


def random_pts_on_sphere(n, mask=None):
    import healpy
    xyz = numpy.random.randn(n, 3)
    l, b = uv2lb(xyz)
    if mask is not None:
        t, p = lb2tp(l, b)
        nside = healpy.npix2nside(len(mask))
        pix = healpy.ang2pix(nside, t, p)
        m = (mask[pix] != 0)
        l, b = l[m], b[m]
    return l, b


def match_radec(r1, d1, r2, d2, rad=1./60./60., nneighbor=0, notself=False):
    # warning: cKDTree has issues if there are large numbers of points
    # at the exact same positions (it takes forever / reaches maximum
    # recursion depth).
    if notself and nneighbor > 0:
        nneighbor += 1
    uv1 = lb2uv(r1, d1)
    uv2 = lb2uv(r2, d2)
    from scipy.spatial import cKDTree
    tree = cKDTree(uv2)
    dub = 2*numpy.sin(numpy.radians(rad)/2)
    if nneighbor > 0:
        d12, m2 = tree.query(uv1, nneighbor, distance_upper_bound=dub)
        if nneighbor > 1:
            m2 = m2.reshape(-1)
            d12 = d12.reshape(-1)

        m1 = numpy.arange(len(r1)*nneighbor, dtype='i4') // nneighbor
        d12 = 2*numpy.arcsin(numpy.clip(d12 / 2, 0, 1))*180/numpy.pi
        m = m2 < len(r2)
    else:
        tree1 = cKDTree(uv1)
        res = tree.query_ball_tree(tree1, dub)
        lens = [len(r) for r in res]
        m2 = numpy.repeat(numpy.arange(len(r2), dtype='i4'), lens)
        if len(m2) > 0:
            m1 = numpy.concatenate([r for r in res if len(r) > 0])
        else:
            m1 = m2.copy()
        d12 = gc_dist(r1[m1], d1[m1], r2[m2], d2[m2])
        m = numpy.ones(len(m1), dtype='bool')
    if notself:
        m = m & (m1 != m2)
    return m1[m], m2[m], d12[m]


def match2d(x1, y1, x2, y2, rad):
    """Find all matches between x1, y1 and x2, y2 within radius rad."""
    from scipy.spatial import cKDTree
    xx1 = numpy.stack([x1, y1], axis=1)
    xx2 = numpy.stack([x2, y2], axis=1)
    tree1 = cKDTree(xx1)
    tree2 = cKDTree(xx2)
    res = tree1.query_ball_tree(tree2, rad)
    lens = [len(r) for r in res]
    m1 = numpy.repeat(numpy.arange(len(x1), dtype='i4'), lens)
    if sum([len(r) for r in res]) == 0:
        m2 = m1.copy()
    else:
        m2 = numpy.concatenate([r for r in res if len(r) > 0])
    d12 = numpy.sqrt(numpy.sum((xx1[m1, :]-xx2[m2, :])**2, axis=1))
    return m1, m2, d12


def unique(obj):
    """Gives an array indexing the unique elements in the sorted list obj.

    Returns the list of indices of the last elements in each group
    of equal-comparing items in obj
    """
    nobj = len(obj)
    if nobj == 0:
        return numpy.zeros(0, dtype='i8')
    if nobj == 1:
        return numpy.zeros(1, dtype='i8')
    out = numpy.zeros(nobj, dtype=numpy.bool)
    out[0:nobj-1] = (obj[0:nobj-1] != obj[1:nobj])
    out[nobj-1] = True
    return numpy.sort(numpy.flatnonzero(out))


def unique_multikey(obj, keys):
    """Gives an array indexing the unique elements in the sorted list obj,
    according to a set of keys.

    Returns the list of indices of the last elements in each group
    of equal-comparing items in obj
    """
    nobj = len(obj[keys[0]])
    if nobj == 0:
        return numpy.zeros(0, dtype='i8')
    if nobj == 1:
        return numpy.zeros(1, dtype='i8')
    out = numpy.zeros(nobj, dtype=numpy.bool)
    for k in keys:
        out[0:nobj-1] |= (obj[k][0:nobj-1] != obj[k][1:nobj])
    out[nobj-1] = True
    return numpy.sort(numpy.flatnonzero(out))


def iqr(dat):
    from scipy.stats.mstats import mquantiles
    quant = mquantiles(dat, (0.25, 0.75))
    return quant[1]-quant[0]


def minmax(v, nan=False):
    v = numpy.asarray(v)
    if nan:
        return numpy.asarray([numpy.nanmin(v), numpy.nanmax(v)])
    return numpy.asarray([numpy.min(v), numpy.max(v)])


class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data, uind=None):
        if uind is None:
            self.uind = unique(data)
        else:
            self.uind = uind.copy()
        self.ind = 0
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.uind)
    def __next__(self):
        if self.ind == len(self.uind):
            raise StopIteration
        if self.ind == 0:
            first = 0
        else:
            first = self.uind[self.ind-1]+1
        last = self.uind[self.ind]+1
        self.ind += 1
        return first, last
    def next(self):
        return self.__next__()


def query_lsd(querystr, db=None, bounds=None, **kw):
    import lsd
    from lsd import DB, bounds as lsdbounds
    if db is None:
        db = os.environ['LSD_DB']
    if not isinstance(db, DB):
        dbob = DB(db)
    else:
        dbob = db
    if bounds is not None:
        bounds = lsdbounds.make_canonical(bounds)
    query = dbob.query(querystr, **kw)
    return query.fetch(bounds=bounds)

def svsol(u,s,vh,b): # N^2 time
    out = numpy.dot(numpy.transpose(u), b)
    s2 = 1./(s + (s == 0))*(s != 0)
    out = numpy.dot(numpy.diag(s2), out)
    out = numpy.dot(numpy.transpose(vh), out)
    return out

def svd_variance(u, s, vh, no_covar=False):
    s2 = 1./(s + (s == 0))*(s != 0)
#    covar = numpy.dot(numpy.dot(numpy.transpose(vh), numpy.diag(s2)),
#                      numpy.transpose(u))
    if no_covar: # computing the covariance matrix is expensive, n^3 time.  if we skip that, only n^2
        return (numpy.array([ numpy.sum(vh.T[i,:]*s2*u.T[:,i]) for i in range(len(s2))]), 
                numpy.nan)
    covar = vh.T*s2.reshape(1,-1)
    covar = numpy.dot(covar, u.T)
    var = numpy.diag(covar)
    return var, covar

def writehdf5(dat, filename, dsname=None, mode='a'):
    if dsname == None:
        dsname = 'default'
    import h5py
    f = h5py.File(filename, mode)
    try:
        f.create_dataset(dsname, data=dat)
        f.close()
    except Exception as e:
        f.close()
        raise e

def readhdf5(filename, dsname=None):
    import h5py
    f = h5py.File(filename, 'r')
    if dsname is None:
        keys = f.keys()
        if len(keys) == 0:
            f.close()
            raise IOError('No data found in file %s' % filename)
        dsname = keys[0]
    try:
        #ds = f.create_dataset(dsname, shape=f[dsname].shape)
        ds = f[dsname]
    except Exception as e:
        print('possible keys:' , f.keys())
        f.close()
        raise e
    dat = ds[:]
    f.close()
    return dat

# stolen from internet, Simon Brunning
def locate(pattern, root=None):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    if root is None:
        root = os.curdir
    for path, dirs, files in os.walk(os.path.abspath(root)):
        files2 = [os.path.join(os.path.relpath(path, start=root), f)
                  for f in files]
        for filename in fnmatch.filter(files2, pattern):
            yield os.path.join(os.path.abspath(root), filename)

# stolen from internet
def congrid(a, newdims, method='linear', center=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    center:
    True - interpolation points are at the centers of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    import scipy.interpolate
    import scipy.ndimage
    import numpy as n
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](center) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append((old[i] - m1) / (newdims[i] - m1) *
                           (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = range(n.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None

# Stolen from MJ
#######################################################################
# Compute percentiles of 2D ndarrays of frequencies
def percentile_freq(v, quantiles, axis=0, rescale=None, catchallbin=True):
    """
    Given an array v, compute quantiles along a given axis
    
    Array v is assumed to be two dimensional.
    """
    if v.dtype != float:
        v = v.astype(float)
    if axis != 0:
        v = v.transpose()

    cs = numpy.cumsum(v, axis=0, dtype='f8')
    c = numpy.zeros((v.shape[0]+1, v.shape[1]))
    c[1:, :] = cs / (cs[-1, :] + (cs[-1, :] == 0))

    # Now find the desired
    #x   = numpy.arange(v.shape[0]+1) - 0.5
    x   = numpy.linspace(0, v.shape[0], v.shape[0]+1)
    res = numpy.empty(numpy.array(quantiles, copy=False).shape +
                      (v.shape[1],), dtype=float)
    norm = x*1e-10
    for k in range(v.shape[1]):
        # Construct interpolation object
        y  = c[:, k] + norm
        # this tiny fudge ensures y is always monotonically increasing
        res[:, k] = numpy.interp(quantiles, y, x)

    if rescale:
        if catchallbin:
            res = rescale[0] + (rescale[1]-rescale[0])*(res-1)/(v.shape[0]-2)
        else:
            res = rescale[0] + (rescale[1] - rescale[0]) * (res/v.shape[0])

    if axis != 0:
        v = v.transpose()

    res[:, cs[-1, :] == 0] = numpy.nan

    return res


# Stolen from MJ
def clipped_stats(v, clip=3, return_mask=False):
    """
    Clipped mean and stdev for a vector.
    
    Computes the median and SIQR for the vector, and excludes all
    samples outside of +/- siqr*(1.349*clip) of the median (i.e., +/-

    clip sigma).  Returns the mean and stdev of the remainder.
    
    Returns
    -------
    mean, stdev: numbers
    The mean and stdev of the clipped vector.
    """
    v = (numpy.atleast_1d(numpy.asarray(v))).flat
    v = v[numpy.isfinite(v)]
    if len(v) == 0:
        return numpy.nan, numpy.nan

    if len(v) == 1:
        return v[0], 0*v[0]
    
    from scipy.stats.mstats import mquantiles
    (ql, med, qu) = mquantiles(v, (0.25, 0.5, 0.75))
    siqr = 0.5*(qu - ql)

    vmin = med - siqr * (1.349*clip)
    vmax = med + siqr * (1.349*clip)
    mask = (vmin <= v) & (v <= vmax) & numpy.isfinite(v)
    v = v[mask].astype(numpy.float64)
    if len(v) <= 0:
        pdb.set_trace()
    ret = numpy.mean(v), numpy.std(v, ddof=1)
    if return_mask:
        ret = ret + (mask,)
    return ret

# stolen from MJ, in numpy 1.6.0 as ravel_multi_index
def ravel_index(coords, shape):
	"""
		Convert a list of n-dimensional indices into a
		1-dimensional one. The inverse of np.unravel_index.
	"""
	idx = 0
	ns = 1
	for c, n in zip(reversed(coords), reversed(shape)):
		idx += c * ns
		ns *= n
	return idx

def fasthist(x, first, num, binsz, weight=None, nchunk=10000000):
    outdims = [n + 2 for n in num]
    dtype = 'u4' if weight is None else weight.dtype
    weight = weight if weight is not None else numpy.ones(len(x[0]),
                                                          dtype='u4')
    out = numpy.zeros(numpy.product(outdims), dtype=dtype)
    i = 0
    while i*nchunk < len(x[0]):
        if len(x[0]) > (i+1)*nchunk:
            xc = [y[i*nchunk:(i+1)*nchunk] for y in x]
            weightc = weight[i*nchunk:(i+1)*nchunk]
        else:
            xc = [y[i*nchunk:] for y in x]
            weightc = weight[i*nchunk:]
        if len(weightc) != len(xc[0]):
            pdb.set_trace()
        out = fasthist_aux(out, outdims, xc, first, num, binsz, weight=weightc)
        # = rather than += because fasthist_aux overwrites out
        i += 1
    out = out.reshape(outdims)
    return out

def fasthist_aux(out, outdims, x, first, num, binsz, weight=None):
    if weight is None:
        weight = numpy.ones(len(x[0]), dtype='u4')
    loc = [numpy.array(numpy.floor((i-f)//sz)+1, dtype='i4')
           for i,f,sz in zip(x, first, binsz)]
    loc = [numpy.clip(loc0, 0, n+1, out=loc0) for loc0, n in zip(loc, num)]
    flatloc = ravel_index(loc, outdims)
    return add_arr_at_ind(out, weight, flatloc)

def make_bins(p, range, bin, npix):
    if bin != None and npix != None:
        print("bin size and number of bins set; ignoring bin size")
        bin = None
    if range is not None and min(range) == max(range):
        raise ValueError('min(range) must not equal max(range)')
    flip = False
    m = numpy.isfinite(p)
    range = (range if range is not None else
             [numpy.min(p[m]), numpy.max(p[m])])
    if range[1] < range[0]:
        flip = True
        range = [range[1], range[0]]
    if bin != None:
        npix = numpy.ceil((range[1]-range[0])/bin)
        npix = npix if npix > 0 else 1
        pts = range[0] + numpy.arange(npix+1)*bin
        if bin == 0:
            pdb.set_trace()
    else:
        npix = (npix if npix is not None else
                numpy.ceil(0.3*numpy.sqrt(len(p))))
        npix = npix if npix > 0 else 1
        pts = numpy.linspace(range[0], range[1], npix+1)
    return range, pts, flip

class Scatter:
    def __init__(self, x, y, xnpix=None, ynpix=None, xrange=None, yrange=None,
                 xbin=None, ybin=None, normalize=True, conditional=True,
                 unserialize=None, weight=None):
        if unserialize is not None:
            savefields = ['hist_all', 'xe_a', 'ye_a', 'norm', 'conditional']
            for f in savefields:
                setattr(self, f, unserialize[0][f])
            self.setup()
            return
        if weight is None:
            useweight = False
            weight = numpy.ones(len(x))
        m = numpy.isfinite(x) & numpy.isfinite(y)
        xrange, xpts, flipx = make_bins(x[m], xrange, xbin, xnpix)
        yrange, ypts, flipy = make_bins(y[m], yrange, ybin, ynpix)
        self.hist_all = fasthist((x[m],y[m]), (xpts[0], ypts[0]),
                                 (len(xpts)-1, len(ypts)-1),
                                 (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                                 weight=weight[m])
        if flipx:
            xpts = xpts[::-1].copy()
            self.hist_all = self.hist_all[::-1,:].copy()
            xrange = [r for r in reversed(xrange)]
        if flipy:
            ypts = ypts[::-1].copy()
            self.hist_all = self.hist_all[:,::-1].copy()
            yrange = [r for r in reversed(yrange)]
        inf = numpy.array([numpy.inf])
        xpts2, ypts2 = [numpy.concatenate([-inf, i, inf])
                        for i in (xpts, ypts)]
        self.xe_a = xpts2.copy()
        self.ye_a = ypts2.copy()
        #self.hist_all, self.xe_a, self.ye_a = \
        #            numpy.histogram2d(x, y, bins=[xpts2, ypts2])
        self.deltx, self.delty = self.xe_a[2]-self.xe_a[1], self.ye_a[2]-self.ye_a[1]
        norm = abs(float(numpy.sum(self.hist_all)*self.deltx*self.delty)) if normalize else 1.
        self.norm = norm
        self.hist_all = self.hist_all/norm
        self.conditional = conditional
        self.setup()

    def setup(self):
        self.xe = self.xe_a[1:-1]
        self.ye = self.ye_a[1:-1]
        self.deltx, self.delty = self.xe[1]-self.xe[0], self.ye[1]-self.ye[0]
        self.hist = self.hist_all[1:-1, 1:-1]
        self.xpts = self.xe[0:-1]+0.5*self.deltx
        self.ypts = self.ye[0:-1]+0.5*self.delty
        zo = numpy.arange(2, dtype='i4')
        self.xrange = self.xpts[-zo]+0.5*self.deltx*(zo*2-1)
        self.yrange = self.ypts[-zo]+0.5*self.delty*(zo*2-1)
        self.coltot = numpy.sum(self.hist_all[1:-1,:], axis=1)
        # 1:-1 to remove the first and last rows in _x_ from the
        # conditional sums. e.g., everything that falls outside of
        # the x limits.  The stuff falling outside of the y limits
        # still counts
        if self.conditional:
            self.q16, self.q50, self.q84 = \
                    percentile_freq(self.hist_all[1:-1,:], [0.16, 0.5, 0.84],
                                    axis=1, rescale=[self.ye[0], self.ye[-1]])

    def show(self, linecolor='k', clipzero=False, **kw):
        hist = self.hist
        if self.conditional:
            coltot = (self.coltot + (self.coltot == 0))
        else:
            coltot = numpy.ones(len(self.coltot))
        dispim = hist / coltot.reshape((len(coltot), 1))
        if clipzero:
            m = dispim == 0
            dispim[m] = numpy.min(dispim[~m])/2
        extent = [self.xe[0], self.xe[-1], self.ye[0], self.ye[-1]]
        if 'nograyscale' not in kw or not kw['nograyscale']:
            pyplot.imshow(dispim.T, extent=extent, interpolation='nearest',
                          origin='lower', aspect='auto', **kw)
        if self.conditional:
            xpts2 = (numpy.repeat(self.xpts, 2) + 
                     numpy.tile(numpy.array([-1.,1.])*0.5*self.deltx,
                                len(self.xpts)))
            pyplot.plot(xpts2, numpy.repeat(self.q16, 2), linecolor,
                        xpts2, numpy.repeat(self.q50, 2), linecolor,
                        xpts2, numpy.repeat(self.q84, 2), linecolor)
        pyplot.xlim(self.xrange)
        pyplot.ylim(self.yrange)

    def serialize(self):
        savefields = ['hist_all', 'xe_a', 'ye_a', 'norm', 'conditional']
        return convert_to_structured_array(self, savefields)

def convert_to_structured_array(ob, fields, getfn=getattr):
    skipzero = [ ]
    dtype = []
    for f in fields:
        val = getfn(ob, f, None)
        if val is None:
            continue
        val = numpy.asarray(val)
        descr = val.dtype.descr
        if numpy.atleast_1d(val) is val:
            #if val.dtype.isbuiltin:
            #    print 'builtin', val.dtype.str
            #    dtype.append((f, val.dtype.str, val.shape))
            #else:
            #    print 'else', [f, val.dtype.descr, val.shape]
            #    dtype += [f, val.dtype.descr, val.shape]
            adddtype = val.dtype.descr
            if len(adddtype) == 1 and adddtype[0][0] == '':
                adddtype = [(f, adddtype[0][1], val.shape)]
            else:
                adddtype = [(f, adddtype, val.shape)]
            if numpy.product(val.shape) != 0:
                dtype += adddtype
            else:
                skipzero.append(f)
        elif val.dtype.fields is not None:
            dtype += [(f, val.dtype.descr)]
        else:
            dtype.append((f, val.dtype.str))
    out = numpy.zeros(1, dtype=dtype)
    for f in fields:
        if f in skipzero:
            print("field %s has zero size, skipping" % f)
            continue
        val = getfn(ob, f, None)
        if val is not None:
            if not numpy.isscalar(out[f][0]):
                val = numpy.array(val)
                assert out[f][0].shape == val.shape, (out[f][0].shape, val.shape)
                assert out[f][0].dtype == val.dtype, (out[f][0].dtype, val.dtype)
            out[f][0] = val
    return out

# numpy.void can't be pickled, so set novoid if you want to get an ndarray
# instead.  These are annoying in that you need an extra [0] index
# everywhere, though.
def dict_to_struct(d, novoid=False):
    getfn = lambda d, f, default=None: d.get(f, default)
    out = convert_to_structured_array(d, d.keys(), getfn)
    if not novoid:
        out = out[0]
    return out

def convert_to_dict(ob):
    out = { }
    for f in ob.dtype.names:
        out[f] = ob[f]
    return out

# stolen from stackoverflow
def join_struct_arrays(arrays):
    newdtype = sum((a.dtype.descr for a in arrays), [])
    names = numpy.array([dt[0] for dt in newdtype])
    s = numpy.argsort(names)
    names = names[s]
    u = unique(names)
    nname = numpy.concatenate([[u[0]+1], u[1:]-u[0:-1]])
    if numpy.any(nname > 1):
        print('Duplicate names.', names[u[nname > 1]])
        raise ValueError('Duplicate column names in table.')
    newrecarray = numpy.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def scatterplot(x, y, xnpix=None, ynpix=None, xrange=None, yrange=None,
                normalize=True, xbin=None, ybin=None, conditional=True,
                log=False, **kw):

    scatter = Scatter(x, y, xnpix=xnpix, ynpix=ynpix, xbin=xbin, ybin=ybin,
                      xrange=xrange, yrange=yrange, normalize=normalize,
                      conditional=conditional)
    if log:
        kw['norm'] = matplotlib.colors.LogNorm()
    scatter.show(**kw)
    return scatter


def make_normal_arr(x):
    if x.flags.contiguous != True or x.flags.owndata != True:
        x = x.copy()
    return x


def pickle_unpickle(x):
    """ Pickles and then unpickles the argument, returning the result.
    Intended to be used to verify that objects pickle successfully."""
    import tempfile
    tf = tempfile.TemporaryFile()
    pickle.dump(x, tf)
    tf.seek(0)
    x = pickle.load(tf)
    tf.close()
    return x


def make_stats_usable(x):
    """ dumps and loads an ipython stats object to render it usable."""
    import pstats
    import tempfile
    tf = tempfile.NamedTemporaryFile()
    x.dump_stats(tf.name)
    stats = pstats.Stats(tf.name)
    tf.close()
    return stats


# given some values val and a dictionary dict, create
# [dict[v] for v in val], but hopefully more quickly?
def convert_val(val, d, default=None):
    s = numpy.argsort(val)
    if len(d) > 0:
        t = type((d.itervalues().next()))
    else:
        t = type(default)
    out = numpy.zeros(len(val), dtype=t)
    for first, last in subslices(val[s]):
        key = val[s[first]]
        if (not d.has_key(key)) and (default is None):
            raise AttributeError('dictionary d missing key /%s/' % str(key))
        out[s[first:last]] = d.get(val[s[first]], default)
    return out

def mag_arr_index(mag, ind):
    return mag[:,ind] if isinstance(ind, (int, long)) else mag[ind]

def ccd(mag, ind1, ind2, ind3, ind4, markersymbol=',', nozero=True,
        norm=matplotlib.colors.LogNorm(),
        xrange=[-1,3], yrange=[-1,3], contour=False, pts=False, **kwargs):
    x = mag_arr_index(mag, ind1) - mag_arr_index(mag, ind2)
    y = mag_arr_index(mag, ind3) - mag_arr_index(mag, ind4)
    if nozero:
        m = (x != 0) & (y != 0)
        x = x[m]
        y = y[m]
    if pts or len(x) < 1000:
        pyplot.plot(x, y, markersymbol, **kwargs)
        pyplot.xlim(xrange) ; pyplot.ylim(yrange)
        return
    if not contour:
        scatterplot(x, y, conditional=False, xrange=xrange, yrange=yrange,
                    norm=norm, **kwargs)
    else:
        contourpts(x, y, xrange=xrange, yrange=yrange, **kwargs)

def cmd(mag, ind1, ind2, ind3, nozero=True, norm=matplotlib.colors.LogNorm(), 
        xrange=[-1,3], yrange=[25,10], contour=False, pts=False, markersymbol=',',
        **kwargs):
    x = mag_arr_index(mag, ind1) - mag_arr_index(mag, ind2)
    y = mag_arr_index(mag, ind3)
    if nozero:
        m = (x != 0) & (y != 0)
        x = x[m]
        y = y[m]
    if pts or len(x) < 1000:
        pyplot.plot(x, y, markersymbol, **kwargs)
        pyplot.xlim(xrange)
        pyplot.ylim(yrange)
        return
    if not contour:
        return scatterplot(x, y, conditional=False, 
                           xrange=xrange, yrange=yrange,
                           norm=norm, **kwargs)
    else:
        return contourpts(x, y, xrange=xrange, yrange=yrange, **kwargs)

def djs_iterstat(dat, invvar=None, sigrej=3., maxiter=10.,
                 prefilter=False, removenan=False, removemask=False):
    """ Straight port of djs_iterstat.pro in idlutils"""
    out = { }
    dat = numpy.atleast_1d(dat)
    if invvar is not None:
        invvar = numpy.atleast_1d(invvar)
        assert len(invvar) == len(dat)
        assert numpy.all(invvar >= 0)
    if removenan:
        keep = numpy.isfinite(dat)
        if invvar is not None:
            keep = keep & numpy.isfinite(invvar)
        dat = dat[keep]
        if invvar is not None:
            invvar = invvar[keep]
        initial_mask = keep
    nan = numpy.nan
    ngood = numpy.sum(invvar > 0) if invvar is not None else len(dat)
    if ngood == 0:
        ntot = len(dat)
        if removenan:
            ntot = len(initial_mask)
        out = {'mean':nan, 'median':nan, 'sigma':nan,
               'mask':numpy.zeros(ntot, dtype='bool'), 'newivar':nan}
        return out
    if ngood == 1:
        val = dat[invvar > 0] if invvar is not None else dat[0]
        out = {'mean':val, 'median':val, 'sigma':0.,
               'mask':numpy.ones(1, dtype='bool'), 'newivar':nan}
        if invvar is not None:
            out['newivar'] = invvar[invvar > 0][0]
        return out
    if invvar is not None:
        mask = invvar > 0
    else:
        mask = numpy.ones_like(dat, dtype='bool')
    if prefilter:
        w = invvar if invvar is not None else None
        quart = weighted_quantile(dat, weight=w, quant=[0.25,0.5, 0.75],
                                  interp=True)
        iqr = quart[2]-quart[0]
        med = quart[1]
        mask = mask & (numpy.abs(dat-med) <= sigrej*iqr)
    if invvar is not None:
        invsig = numpy.sqrt(invvar)
        fmean = numpy.sum(dat*invvar*mask)/numpy.sum(invvar*mask)
    else:
        fmean = numpy.sum(dat*mask)/numpy.sum(mask)
    fsig = numpy.sqrt(numpy.sum((dat-fmean)**2.*mask)/(ngood-1))
    iiter = 1
    savemask = mask

    nlast = -1
    while iiter < maxiter and nlast != ngood and ngood >= 2:
        nlast = ngood
        iiter += 1
        if invvar is not None:
            mask = (numpy.abs(dat-fmean)*invsig < sigrej) & (invvar > 0)
        else:
            mask = numpy.abs(dat-fmean) < sigrej*fsig
        ngood = numpy.sum(mask)
        if ngood >= 2:
            if invvar is not None:
                fmean = numpy.sum(dat*invvar*mask) / numpy.sum(invvar*mask)
            else:
                fmean = numpy.sum(dat*mask) / ngood
            fsig = numpy.sqrt(numpy.sum((dat-fmean)**2.*mask) / (ngood-1))
            savemask = mask
    fmedian = numpy.median(dat[savemask])
    newivar = nan
    if invvar is not None:
        newivar = numpy.sum(invvar*savemask)
    if removenan:
        initial_mask[initial_mask] = savemask
        savemask = initial_mask
    ret = {'mean':fmean, 'median':fmedian, 'sigma':fsig,
           'mask':savemask != 0, 'newivar':newivar}
    if removemask:
        ret.pop('mask')
    return ret

def check_sorted(x):
    if len(x) <= 1:
        return True
    return numpy.all(x[1:len(x)] >= x[0:-1])

def solve_lstsq(aa, bb, ivar, svdthresh=None, return_covar=False):
    d, t = numpy.dot, numpy.transpose
    atcinvb = d(t(aa), ivar*bb)
    atcinva = d(t(aa), ivar.reshape((len(ivar), 1))*aa)
    u,s,vh = numpy.linalg.svd(atcinva)
    if svdthresh is not None:
        s[s < svdthresh] = 0.
    par = svsol(u,s,vh,atcinvb)
    var, covar = svd_variance(u, s, vh)
    ret = (par, var)
    if return_covar:
        ret = ret + (covar,)
    return ret

def solve_lstsq_covar(aa, bb, icvar, svdthresh=None, return_covar=False):
    d, t = numpy.dot, numpy.transpose
    atcinvb = d(t(aa), d(icvar,bb))
    atcinva = d(t(aa), d(icvar,aa))
    u,s,vh = numpy.linalg.svd(atcinva)
    if svdthresh is not None:
        s[s < svdthresh] = 0.
    par = svsol(u,s,vh,atcinvb)
    var, covar = svd_variance(u, s, vh)
    ret = (par, var)
    if return_covar:
        ret = ret + (covar,)
    return ret

def polyfit(x, y, deg, ivar=None, return_covar=False):
    aa = numpy.zeros((len(x), deg+1))
    for i in range(deg+1):
        aa[:,i] = x**(deg-i)
    if ivar is None:
        ivar = numpy.ones_like(y)
    m = numpy.isfinite(ivar)
    m[m] &= ivar[m] > 0
    ivar = ivar.copy()
    ivar[~m] = 0
    return solve_lstsq(aa, y, ivar, return_covar=return_covar)

def poly_iter(x, y, deg, yerr=None, sigrej=3., niter=10, return_covar=False,
              return_mask=False):
    m = numpy.ones_like(x, dtype='bool')
    if yerr is None:
        invvar = None
    else:
        invvar = 1./yerr**2
    for i in range(niter):
        iv = invvar[m] if invvar is not None else None
        sol = polyfit(x[m], y[m], deg, ivar=iv, return_covar=return_covar)
        p = sol[0]
        if numpy.sum(m) <= 1:
            break
        res = y - numpy.polyval(p, x)
        if invvar is None:
            stats = djs_iterstat(res[m], invvar=iv, sigrej=sigrej,
                                 prefilter=True)
            mn, sd = clipped_stats(res[m], clip=sigrej)
            m2 = numpy.abs(res - mn)/(sd+(sd == 0)) < sigrej
        else:
            chi = res*invvar**0.5
            stats = djs_iterstat(chi[m], prefilter=True)
            mn, sd = clipped_stats(chi[m], clip=sigrej)
            m2 = numpy.abs(chi) < sigrej
        if numpy.all(m2 == m):
            break
        else:
            m = m2
    if return_mask:
        sol = sol + (m,)
    return sol

def weighted_quantile(x, weight=None, quant=[0.25, 0.5, 0.75], interp=False):
    if weight is None:
        weight = numpy.ones_like(x)
    weight = weight / numpy.float(numpy.sum(weight))
    s = numpy.argsort(x)
    weight = weight[s]
    if interp:
        weight1 = numpy.cumsum(weight)
        weight2 = 1.-(numpy.cumsum(weight[::-1])[::-1])
        weight = (weight1 + weight2)/2.
        pos = numpy.interp(quant, weight, numpy.arange(len(weight)))
        quant = numpy.interp(pos, numpy.arange(len(weight)), x[s])
    else:
        weight = numpy.cumsum(weight)
        weight /= weight[-1]
        pos = numpy.searchsorted(weight, quant)
        quant = numpy.zeros(len(quant))
        quant[pos <= 0] = x[s[0]]
        quant[pos >= len(x)] = x[s[-1]]
        m = (pos > 0) & (pos < len(x))
        quant[m] = x[s[pos[m]]]
    return quant

def lb2tp(l, b):
    return (90.-b)*numpy.pi/180., l*numpy.pi/180.

def tp2lb(t, p):
    return p*180./numpy.pi % 360., 90.-t*180./numpy.pi

def tp2uv(t, p):
    z = numpy.cos(t)
    x = numpy.cos(p)*numpy.sin(t)
    y = numpy.sin(p)*numpy.sin(t)
    return numpy.concatenate([q[...,numpy.newaxis] for q in (x, y, z)],
                             axis=-1)
    #return numpy.vstack([x, y, z]).transpose().copy()
    
def lb2uv(r, d):
    return tp2uv(*lb2tp(r, d))

def uv2tp(uv):
    norm = numpy.sqrt(numpy.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = numpy.arccos(uv[:,2])
    p = numpy.arctan2(uv[:,1], uv[:,0])
    return t, p

def xyz2tp(x, y, z):
    norm = numpy.sqrt(x**2+y**2+z**2)
    t = numpy.arccos(z/norm)
    p = numpy.arctan2(y/norm, x/norm)
    return t, p

def uv2lb(uv):
    return tp2lb(*uv2tp(uv))

def xyz2lb(x, y, z):
    return tp2lb(*xyz2tp(x, y, z))

def lbr2xyz_galactic(l, b, re, r0=8.5):
    """This seems to be my made up right-handed coordinate system.
    It's not the usual choice.  x increases from the GC to the Earth,
    y increases toward l=-90, and z increases toward the NGC.  Galactic
    UVW systems all have y increasing toward l=90 (direction of Galactic
    rotation).  They can alternatively be left handed and have U increasing
    toward the Galactic anticenter."""
    l, b = numpy.radians(l), numpy.radians(b)
    z = re * numpy.sin(b)
    x = r0 - re*numpy.cos(l)*numpy.cos(b)
    y = -re*numpy.sin(l)*numpy.cos(b)
    return x, y, z

def xyz_galactic2lbr(x, y, z, r0=8.5):
    """See note about this coordinate system in lbr2xyz_galactic."""
    xe = r0-x
    re = numpy.sqrt(xe**2+y**2+z**2)
    b = numpy.degrees(numpy.arcsin(z / re))
    l = numpy.degrees(numpy.arctan2(-y, xe)) % 360.
    return l, b, re


def xyz2rphiz(x, y, z):
    r = numpy.sqrt(x**2+y**2)
    phi = numpy.degrees(numpy.arctan2(y, x))
    return r, phi, z


# should write a lbr2uvw for the right handed coordinate system.
# galpy apparently uses the left handed coordinate system, though.

def lbr2uvw_galactic(l, b, re):
    """Right handed, U increasing toward the GC, V increasing toward l=90,
    W increasing toward the NGC.  Origin at the earth."""

    l, b = numpy.radians(l), numpy.radians(b)
    w = re*numpy.sin(b)
    u = re*numpy.cos(l)*numpy.cos(b)
    v = re*numpy.sin(l)*numpy.cos(b)
    # very familiar!
    return u, v, w


def uvw_galactic2lbr(u, v, w):
    re = numpy.sqrt(u**2+v**2+w**2)
    b = numpy.degrees(numpy.arcsin(w/re))
    l = numpy.degrees(numpy.arctan2(v, u)) % 360.
    return l, b, re


def healgen(nside):
    import healpy
    return healpy.pix2ang(nside, numpy.arange(12*nside**2))

def healgen_lb(nside):
    return tp2lb(*healgen(nside))

def heal_rebin(map, nside, ring=True):
    import healpy
    if ring:
        map = healpy.reorder(map, r2n=True)
    if map.dtype.name == 'bool':
        map = map.astype('f4')
    nside_orig = healpy.get_nside(map)
    if nside_orig % nside != 0:
        raise ValueError('nside must divide nside_orig')
    binfac = int((nside_orig / nside)**2)
    assert binfac * 12*nside**2 == len(map), 'Inconsistent sizes in heal_rebin.'
    newmap = map.copy()
    newmap = map.reshape((-1, binfac))
    newmap = numpy.sum(newmap, axis=1)
    if ring:
        newmap = healpy.reorder(newmap, n2r=True)
    return newmap / binfac

def heal_rebin_mask(map, nside, mask, ring=True, nanbad=False):
    newmap = heal_rebin(map*mask, nside, ring=ring)
    newmask = heal_rebin(mask, nside, ring=ring)
    out = newmap/(newmask + (newmask == 0))
    if nanbad:
        out[newmask == 0] = numpy.nan
    return out

def heal2cart(heal, interp=True, return_pts=False):
    import healpy
    nside = healpy.get_nside(heal)#*(2 if interp else 1)
    owidth = 8*nside
    oheight = 4*nside-1
    dm,rm = numpy.mgrid[0:oheight,0:owidth]
    rm = 360.-(rm+0.5) / float(owidth) * 360.
    dm = -90. + (dm+0.5) / float(oheight) * 180.
    t, p = lb2tp(rm.ravel(), dm.ravel())
    if interp:
        map = healpy.get_interp_val(heal, t, p)
    else:
        pix = healpy.ang2pix(nside, t, p)
        map = heal[pix]
    map = map.reshape((oheight, owidth))
    if return_pts:
        map = (map, numpy.sort(numpy.unique(rm)), numpy.sort(numpy.unique(dm)))
    return map


def imshow(im, xpts=None, ypts=None, xrange=None, yrange=None, range=None,
           min=None, max=None, mask_nan=False,
           interp_healpy=True, log=False, center_gal=False, center_l=None, color=False,
           return_handler=False, contour=None,
           **kwargs):
    if xpts is not None and ypts is not None:
        dx = numpy.median(xpts[1:]-xpts[:-1])
        dy = numpy.median(ypts[1:]-ypts[:-1])
        kwargs['extent'] = [xpts[0]-dx/2., xpts[-1]+dx/2.,
                            ypts[0]-dy/2., ypts[-1]+dy/2.]
        if len(xpts) != im.shape[0] or len(ypts) != im.shape[1]:
            print('Warning: mismatch between xpts, ypts and im.shape')
        if not color:
            im = im.T
        else:
            im = numpy.transpose(im, axes=[1, 0, 2])
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    #kwargs['origin'] = 'upper'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'binary'
    if isinstance(kwargs['cmap'], str):
        from copy import deepcopy
        kwargs['cmap'] = deepcopy(matplotlib.cm.get_cmap(kwargs['cmap']))
    oneim = im if not color else im[:,0]
    import healpy
    if (len(oneim.shape) == 1) and healpy.isnpixok(len(oneim)):
        if not color:
            im = heal2cart(im, interp=interp_healpy)
        else:
            outim = None
            ncolor = im.shape[-1]
            for i in numpy.arange(ncolor, dtype='i4'):
                oneim = heal2cart(im[:,i], interp=interp_healpy)
                if outim is None:
                    outim = numpy.zeros(oneim.shape+(ncolor,))
                outim[:,:,i] = oneim
            im = outim
        if center_gal and center_l is not None:
            raise ValueError('May only set one of center_gal, center_l')
        if center_l is None:
            if not center_gal:
                center_l = 180
            else:
                center_l = 0
        if center_l == 180:
            kwargs['extent'] = ((360, 0, -90, 90))
        else:
            left_l = (center_l - 180) % 360.
            # print left_l
            if not color:
                im = numpy.roll(im, numpy.int((im.shape[1]*left_l)/360), axis=1)
            else:
                ncolor = im.shape[-1]
                for i in numpy.arange(ncolor,dtype='i4'):
                    im[:,:,i] = numpy.roll(im[:,:,i], numpy.int(im.shape[1]*left_l/360), axis=1)
            kwargs['extent'] = (left_l, left_l - 360., -90, 90)
    if range is not None:
        if min is not None or max is not None:
            raise ValueError('Can not set both range and min/max')
        kwargs['vmin'] = range[0]
        kwargs['vmax'] = range[1]
    if min is not None:
        kwargs['vmin'] = min
    if max is not None:
        kwargs['vmax'] = max
    if log:
        kwargs['norm'] = matplotlib.colors.LogNorm()
    if mask_nan:
        #im = numpy.ma.array(im, mask=numpy.isnan(im))
        kwargs['cmap'].set_bad('lightblue', 1)
    else:
        kwargs['cmap'].set_bad('lightblue', 0)
    updatecolorscale = kwargs.pop('updatecolorscale', False)
    if contour is None:
        out = pyplot.imshow(im, **kwargs)
    else:
        out = pyplot.contour(im, levels=contour, **kwargs)
        pyplot.xlim(kwargs['extent'][0:2])
        pyplot.ylim(kwargs['extent'][2:4])
    if xrange is not None:
        pyplot.xlim(xrange)
    if yrange is not None:
        pyplot.ylim(yrange)
    if contour is None:
        events = EventHandlerImshow(pyplot.gcf(), out,
                                    updatecolorscale=updatecolorscale)
    if return_handler:
        out = (out, events)
    return out

class EventHandlerImshow():
    def __init__(self, fig, im1, updatecolorscale=False):
        self.button_pressed = False
        self.fig = fig
        self.im1 = im1
        self.xfrac, self.yfrac = 0.5, 0.5
        self.min, self.max = (numpy.nanmin(numpy.array(self.im1._A)),
                              numpy.nanmax(numpy.array(self.im1._A)))
        self.fig.canvas.mpl_connect('button_press_event',
                                    lambda x: self.handle_press(x))
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    lambda x: self.handle_motion(x))
        self.fig.canvas.mpl_connect('button_release_event',
                                    lambda x: self.handle_release(x))
        self.updatecolorscale = updatecolorscale

    def handle_press(self, event):
        if self.fig.canvas.manager.toolbar.mode != "" or event.inaxes != self.im1.axes:
            return
        if event.button == 1:
            self.update_color_scale(event.xdata, event.ydata)
            self.button_pressed = True

    def handle_motion(self, event):
        if self.fig.canvas.manager.toolbar.mode != "" or event.inaxes != self.im1.axes:
            return
        if self.button_pressed:
            self.update_color_scale(event.xdata, event.ydata)

    def handle_release(self, event):
        if self.fig.canvas.manager.toolbar.mode != "" or event.inaxes != self.im1.axes:
            return
        if event.button == 1:
            self.button_pressed = False

    def update_color_scale(self, x=None, y=None):
        if x is not None and y is not None:
            xl, yl = self.im1.axes.get_xlim(), self.im1.axes.get_ylim()
            self.xfrac = (x-xl[0]) / float(xl[1]-xl[0])
            self.yfrac = (y-yl[0]) / float(yl[1]-yl[0])
        xfrac = self.xfrac
        yfrac = self.yfrac
        if self.updatecolorscale:
            self.min, self.max = (numpy.nanmin(numpy.array(self.im1._A)),
                                  numpy.nanmax(numpy.array(self.im1._A)))
        if isinstance(self.im1.norm, matplotlib.colors.LogNorm):
            if self.min <= 0:
                self.min = 1 # dumb
            self.min, self.max = numpy.log10([self.min, self.max])
        centercolor = self.min + (self.max - self.min)*xfrac
        if yfrac > 0.5:
            rangecolor = (self.max-self.min)* 10**((yfrac-0.5)/0.5)*0.5
        else:
            rangecolor = (self.max-self.min)*yfrac
        newrange = [centercolor-rangecolor, centercolor+rangecolor]
        if isinstance(self.im1.norm, matplotlib.colors.LogNorm):
            newrange = [10.**x for x in newrange]
        self.im1.set_clim(newrange)
        self.fig.canvas.draw()

def contourpts(x, y, xnpix=None, ynpix=None, xrange=None, yrange=None,
                normalize=True, xbin=None, ybin=None, log=False,
                levels=None, nlevels=6, logmin=0, symbol=',', minlevel=1,
               logspace=None, nopoints=False,
               **kw):
     sc = Scatter(x, y, xnpix=xnpix, ynpix=ynpix, xbin=xbin, ybin=ybin,
                  xrange=xrange, yrange=yrange, normalize=normalize,
                  conditional=False)
     if levels is None:
         maximage = numpy.float(numpy.max(sc.hist*sc.norm))
         if log:
             if logspace is None:
                 levels = 10.**(logmin + (numpy.arange(nlevels)+1)*
                                (numpy.log10(maximage)-logmin)/nlevels)/sc.norm
             else:
                 levels = 10.**(numpy.log10(maximage)-logspace*numpy.arange(nlevels))[::-1]/sc.norm
         else:
             levels = (numpy.arange(nlevels)+minlevel)*maximage/nlevels

     if 'colors' not in kw:
         kw['colors'] = 'black'
     if 'color' not in kw:
         kw['color'] = 'black'
     pyplot.contour(sc.hist.T, extent=[sc.xe[0], sc.xe[-1],
                                       sc.ye[0], sc.ye[-1]],
                       interpolation='nearest', origin='lower', aspect='auto',
                       levels=levels, **kw)
     flipx = sc.deltx < 0
     flipy = sc.delty < 0
     xloc = numpy.array(numpy.floor((x-sc.xe[0]) #[0 if not flipx else -1])
                                    /sc.deltx), dtype='i4')
     yloc = numpy.array(numpy.floor((y-sc.ye[0]) # if not flipy else -1])
                                    /sc.delty), dtype='i4')
     m = ((xloc >= 0) & (xloc < len(sc.xpts)) &
          (yloc >= 0) & (yloc < len(sc.ypts)))
     m[m] &= sc.hist[xloc[m], yloc[m]] < min(levels)
     kw.pop('colors', None)
     if not nopoints:
         pyplot.plot(x[m], y[m], symbol, **kw)
     pyplot.xlim(sc.xrange)
     pyplot.ylim(sc.yrange)

def match(a, b):
    sa = numpy.argsort(a)
    sb = numpy.argsort(b)
    ua = unique(a[sa])
    ub = unique(b[sb])
    if len(ua) != len(a):# or len(ub) != len(b):
        raise ValueError('All keys in a must be unique.')
    ind = numpy.searchsorted(a[sa], b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], numpy.flatnonzero(m)

def match_sorted_unique(a, b):
    ind = numpy.searchsorted(a, b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[ind[m]] == b[m]
    m[m] &= matches
    return ind[m], numpy.flatnonzero(m)

def setup_tex():
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='serif')
    rc('font', serif=['Computer Modern'])

def setup_print(size=None, keys=None, scalefont=1., **kw):
    params = {'backend': 'ps',
              'axes.labelsize': 12*scalefont,
              'font.size':12*scalefont,
              'legend.fontsize': 10*scalefont,
              'xtick.labelsize': 10*scalefont,
              'ytick.labelsize': 10*scalefont,
              'axes.titlesize':18*scalefont,
              }
    for key in kw:
        params[key] = kw[key]
    if keys is not None:
        for key in keys:
            params[key] = keys[key]
    oldparams = dict(pyplot.rcParams.items())
    pyplot.rcParams.update(params)
    if size is not None:
        pyplot.gcf().set_size_inches(*size, forward=True)
    return oldparams

def arrow(x, y, dx, dy, arrowstyle='->', mutation_scale=30, **kw):
    add_patch = pyplot.gca().add_patch
    FancyArrow = matplotlib.patches.FancyArrowPatch
    return add_patch(FancyArrow((x,y),(x+dx,y+dy), arrowstyle=arrowstyle,
                                mutation_scale=mutation_scale, **kw))
    
def rebin(a, *args):
    """ Stolen from internet
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and
    4 rows can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = numpy.asarray(shape)/numpy.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)]
    return eval(''.join(evList))

def data2map(data, l, b, weight=None, nside=512, finiteonly=True):
    if weight is None:
        weight = numpy.ones(len(data))
    if finiteonly:
        m = (numpy.isfinite(data) & numpy.isfinite(weight) &
             numpy.isfinite(l) & numpy.isfinite(b))
        data = data[m]
        l = l[m]
        b = b[m]
        weight = weight[m]
    t, p = lb2tp(l, b)
    import healpy
    pix = healpy.ang2pix(nside, t, p)
    out = numpy.zeros(12*nside**2)
    wmap = numpy.zeros_like(out)
    out = add_arr_at_ind(out, weight*data, pix)
    wmap = add_arr_at_ind(wmap, weight, pix)
    out = out / (wmap + (wmap == 0))
    return out, wmap


def paint_map(r, d, dat, rad, weight=None, nside=512):
    import healpy
    npix = 12*nside**2
    vec = healpy.ang2vec(*lb2tp(r, d))
    map = numpy.zeros(npix)
    wmap = numpy.zeros(npix)
    if weight is None:
        weight = numpy.ones(len(dat), dtype='i4')
    for v, d, w in zip(vec, dat, weight):
        pix = healpy.query_disc(nside, v, rad*numpy.pi/180.)
        map[pix] += d
        wmap[pix] += w
    map = map / (wmap + (wmap == 0))
    return map, wmap


def bindatan(coord, dat, weight=None, npix=None, ranges=None, bins=None):
    m = numpy.logical_and.reduce([numpy.isfinite(c) for c in coord])
    ranges0 = []
    pts = []
    flips = []
    for i in range(len(coord)):
        trange = None if ranges is None else ranges[i]
        tbin = None if bins is None else bins[i]
        tnpix = None if npix is None else npix[i]
        trange, tpts, tflip = make_bins(coord[i], trange, tbin, tnpix)
        ranges0.append(trange)
        pts.append(tpts)
        flips.append(tflip)
    bins = [numpy.median(tpts[1:]-tpts[0:-1]) for tpts in pts]
    ranges = ranges0

    if weight is None:
        weight = numpy.ones(len(dat))
    hist_all = fasthist(coord, [tpts[0] for tpts in pts],
                        [len(tpts)-1 for tpts in pts],
                        [tpts[1]-tpts[0] for tpts in pts],
                        weight=weight*dat)
    whist_all = fasthist(coord, [tpts[0] for tpts in pts],
                         [len(tpts)-1 for tpts in pts],
                         [tpts[1]-tpts[0] for tpts in pts],
                         weight=weight)
    for i in range(len(flips)):
        if flips[i]:
            pts[i] = pts[i][::-1].copy()
            hist_all = numpy.flip(hist_all, i)
            whist_all = numpy.flip(whist_all, i)
            ranges[i] = [r for r in reversed(ranges[i])]
    ptscen = [(tpts[:-1]+tpts[1:])/2. for tpts in pts]
    return (hist_all/(whist_all + (whist_all == 0)), whist_all.copy(),
            ptscen)


def bindata(x, y, dat, weight=None, xnpix=None, ynpix=None, xrange=None,
            yrange=None, xbin=None, ybin=None):
    # could replace with call to bindatan
    m = numpy.isfinite(x) & numpy.isfinite(y)
    xrange, xpts, flipx = make_bins(x[m], xrange, xbin, xnpix)
    yrange, ypts, flipy = make_bins(y[m], yrange, ybin, ynpix)
    xbin = numpy.median(xpts[1:]-xpts[0:-1])
    ybin = numpy.median(ypts[1:]-ypts[0:-1])

    if weight is None:
        weight = numpy.ones(len(dat))
    hist_all = fasthist((x,y), (xpts[0], ypts[0]),
                        (len(xpts)-1, len(ypts)-1),
                        (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                        weight=weight*dat)
    whist_all = fasthist((x,y), (xpts[0], ypts[0]),
                         (len(xpts)-1, len(ypts)-1),
                         (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                         weight=weight)
    if flipx:
        xpts = xpts[::-1].copy()
        hist_all = hist_all[::-1,:].copy()
        whist_all = whist_all[::-1,:].copy()
        xrange = [r for r in reversed(xrange)]
    if flipy:
        ypts = ypts[::-1].copy()
        hist_all = hist_all[:,::-1].copy()
        whist_all = whist_all[:,::-1].copy()
        yrange = [r for r in reversed(yrange)]
    return (hist_all/(whist_all + (whist_all == 0)), whist_all,
            (xpts[:-1]+xpts[1:])/2., (ypts[:-1]+ypts[1:])/2.)

def showbindata(x, y, dat, xrange=None, yrange=None, min=None, max=None, 
                showweight=False, log=False, **kw):
    im, wim, xpts, ypts = bindata(x, y, dat, xrange=xrange, yrange=yrange,
                                  **kw)
    if showweight:
        im = wim
    imshow(im[1:-1,1:-1], xpts, ypts, xrange=xrange, yrange=yrange, 
           min=min, max=max, log=log)


def histpts(x, y, dat, weight=None, xnpix=None, ynpix=None, xrange=None,
            yrange=None, xbin=None, ybin=None, pointcut=1, cmap='jet', log=False,
            vmin=None, vmax=None, showpts=True, contour=True, nlevels=6, 
            logcontour=False, logspace=None, logmin=0, levels=None, minlevel=1, 
            mask_nan=False, colors='black', **kw):
    m = numpy.isfinite(x) & numpy.isfinite(y)
    xrange, xpts, flipx = make_bins(x[m], xrange, xbin, xnpix)
    yrange, ypts, flipy = make_bins(y[m], yrange, ybin, ynpix)
    xbin = numpy.median(xpts[1:]-xpts[0:-1])
    ybin = numpy.median(ypts[1:]-ypts[0:-1])
    if weight is None:
        weight = numpy.ones(len(dat))
    hist_all = fasthist((x,y), (xpts[0], ypts[0]),
                        (len(xpts)-1, len(ypts)-1),
                        (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                        weight=weight*dat)
    whist_all = fasthist((x,y), (xpts[0], ypts[0]),
                         (len(xpts)-1, len(ypts)-1),
                         (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                         weight=weight)
    count_hist = fasthist((x, y), (xpts[0], ypts[0]),
                          (len(xpts)-1, len(ypts)-1),
                          (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                          weight=numpy.ones(len(dat)))

    loc = [numpy.array(numpy.floor((i-f)//sz)+1, dtype='i4')
           for i,f,sz in zip((x, y), (xpts[0], ypts[0]), (xpts[1]-xpts[0], ypts[1]-ypts[0]))]
    loc = [numpy.clip(loc0, 0, n+1, out=loc0) for loc0, n in 
           zip(loc, (len(xpts)-1, len(ypts)-1))]
    
    
    # uhh, dumbest thing is max one point per bin
    m = count_hist <= pointcut
    hist_all = hist_all / (whist_all + (whist_all == 0.))
    hist_all[m] = numpy.nan
    if vmin is None:
        vmin = numpy.nanmin(hist_all)
    if vmax is None:
        vmax = numpy.nanmax(hist_all)

    if flipx:
        xpts = xpts[::-1].copy()
        hist_all = hist_all[::-1,:].copy()
        count_hist = count_hist[::-1,:].copy()
        xrange = [r for r in reversed(xrange)]
    if flipy:
        ypts = ypts[::-1].copy()
        hist_all = hist_all[:,::-1].copy()
        count_hist = count_hist[:,::-1].copy()
        yrange = [r for r in reversed(yrange)]

    ret = imshow(hist_all[1:-1,1:-1], xpts[:-1]+xbin/2., ypts[:-1]+ybin/2., xrange=xrange, yrange=yrange, cmap=cmap, log=log, vmin=vmin, vmax=vmax, origin='lower', mask_nan=mask_nan)
    # pyplot.draw()
    if showpts:
        # need to find the points that were in a bin with only 1 point.
        m = ~numpy.isfinite(hist_all[loc])
        ret = pyplot.scatter(x[m], y[m], c=dat[m], edgecolor='none', vmin=vmin, vmax=vmax, cmap=cmap, **kw)
    if contour:
        if levels is None:
            maximage = numpy.float(numpy.max(count_hist[1:-1,1:-1]))
            if logcontour:
                if logspace is None:
                    levels = 10.**(logmin + (numpy.arange(nlevels)+1)*
                                   (numpy.log10(maximage)-logmin)/nlevels)
                else:
                    levels = 10.**(numpy.log10(maximage)-logspace*numpy.arange(nlevels))
            else:
                levels = (numpy.arange(nlevels)+minlevel)*maximage/nlevels

        pyplot.contour(count_hist[1:-1,1:-1].T, 
                       extent=[xpts[0], xpts[-1], ypts[0], ypts[-1]],
                       colors=colors, levels=levels)
    return ret

def cg_to_fits(cg, fixub=True):
    dtype = cg.dtype
    for col in dtype.descr:
        colname = col[0]
        format = col[1]
        if format.find('O') != -1:
            cg.drop_column(colname)
            print('Warning: dropping column %s because FITS does not support objects.' % colname)
            continue
        if not fixub:
            continue
        uloc = format.find('u')
        if uloc == -1:
            uloc = format.find('b')
        if uloc == -1:
            continue
        orig = cg[colname]
        newbytes = format[uloc+1:]
        if len(newbytes) == 1 and int(newbytes) < 4:
            newbytes = '4'
        newformat = format[:uloc]+'i'+newbytes
        new = numpy.array(orig, newformat)
        from lsd import colgroup
        if isinstance(cg, colgroup.ColGroup):
            cg.drop_column(colname)
            cg.add_column(colname, new)
        else:
            from matplotlib.mlab import rec_append_fields, rec_drop_fields
            # slow for big structures; should do all the columns at once
            # for the ndarray case.
            cg = rec_append_fields(rec_drop_fields(cg, [colname]), [colname],
                                   [new])
    return cg


def ndarraytofits(nd):
    dtype = nd.dtype
    newdtype = []
    for col in dtype.descr:
        col = list(col)
        format = col[1]
        uloc = format.find('u')
        if uloc != -1:
            col[1] = format[:uloc] + 'i' + format[uloc+1:]
        uloc = format.find('b')
        if uloc != -1:
            col[1] = format[:uloc] + 'i' + format[uloc+1:]
        newdtype.append(tuple(col))
    newnd = numpy.zeros(len(nd), dtype=newdtype)
    for col in nd.dtype.names:
        newnd[col] = nd[col]
    return newnd


def fitstondarray(fits, lower=False):
    from copy import deepcopy
    nd = numpy.zeros(len(fits), fits.dtype)
    for f in fits.dtype.names:
        nd[f] = fits[f]
    if lower:
        names = list(deepcopy(nd.dtype.names))
        names = [n.lower() for n in names]
        nd.dtype.names = names
    return nd


def stirling_approx(n):
    return n*numpy.log(n) - n + numpy.log(n)/2. + numpy.log(2*numpy.pi)/2.


def mjd2lst(mjd, lng):
    """ Stolen from ct2lst.pro in IDL astrolib.
    Returns the local sidereal time at a given MJD and longitude. """

    mjdstart = 2400000.5
    jd = mjd + mjdstart
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0 ]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0/36525.
    theta = c[0] + (c[1] * t0) + t**2*(c[2] - t/ c[3] )
    lst = (theta + lng)/15.
    lst = lst % 24.
    return lst


# PS1 lat/lon: 20.71552, -156.169
def rdllmjd2altaz(r, d, lat, lng, mjd, precess=True):
    if precess:
        import precess as precessmod
        jd2000 = 2451545.0
        mjdstart = 2400000.5
        r, d = precessmod.precess(r, d, 2000., 2000.+(mjd + mjdstart - jd2000)/365.25)
    lst = mjd2lst(mjd, lng)
    ha = lst*360./24 - r
    return hadec2altaz(ha, d, lat)


def hadec2altaz(ha, dec, lat):
    d2r = numpy.pi/180.
    # hadec2altaz.pro in idlutils
    sin, cos = numpy.sin, numpy.cos
    sh, ch = sin(ha*d2r),  cos(ha*d2r)
    sd, cd = sin(dec*d2r), cos(dec*d2r)
    sl, cl = sin(lat*d2r), cos(lat*d2r)
    x = - ch * cd * sl + sd * cl
    y = - sh * cd
    z = ch * cd * cl + sd * sl
    r = numpy.sqrt(x**2 + y**2)
    az = numpy.arctan2(y, x) / d2r
    alt = (numpy.arctan2(z, r) / d2r) % 360.
    return alt, az

def alt2airmass(alt):
    # Pickering (2002) according to Wikipedia?
    #h = alt #90-alt
    #return 1./(numpy.sin(numpy.radians(h + 244 / (165 + 47*h**1.1))))
    # disagrees with sec(z) by 0.001 - 0.002 near zenith, which is the most important part anyway.
    # probably ~1% better at airmass 3, and rapidly better after that---but who cares?
    return 1./numpy.cos(numpy.radians(90-alt))

def interpolate_from(fits, r, d, wcs=None, im=None):
    if wcs is None:
        import pywcs
        from astropy.io import fits
        h = fits.getheader(fits)
        wcs = pywcs.WCS(h)
    if im is None:
        from astropy.io import fits
        im = fits.getdata(fits)
    ims = im.squeeze()
    pix = wcs.wcs_sky2pix(numpy.array([r, d,
                                       numpy.ones(len(r)),
                                       numpy.ones(len(r))]).transpose(), 0)
    from scipy.ndimage import map_coordinates
    val = map_coordinates(ims, (pix[:,0:2])[:,::-1].transpose(),
                 cval=numpy.nan, order=1)
    return val


def write_file_in_chunks(dat, filename, chunksize, breakkey=None):
    from astropy.io import fits
    nbytes = len(dat)*dat.dtype.itemsize
    if breakkey is not None:
        s = numpy.argsort(dat[breakkey])
        dat = dat[s]
    nrecperfile = chunksize / dat.dtype.itemsize
    i = 0
    dotpos = filename[::-1].find('.')
    if dotpos != -1:
        filestart, fileend = filename[0:-dotpos-1], filename[-dotpos-1:]
    else:
        filestart, fileend = filename, ''
    if breakkey is None:
        while i*nrecperfile < len(dat):
            filename = filestart + (('_%d' % i) if i != 0 else '')+fileend
            print(filename, i*nrecperfile, (i+1)*nrecperfile)
            fits.writeto(filename, dat[i*nrecperfile:(i+1)*nrecperfile])
            i += 1
    else:
        first = 0
        last = 0
        i = 0
        for f,l in subslices(dat[breakkey]):
            last += l-f
            if last-first > nrecperfile or (last != first and l == len(dat)):
                filename = filestart + (('_%d' % i) if i != 0 else '')+fileend
                print(filename, first, last, last-first)
                fits.writeto(filename, dat[first:last])
                first = last
                i += 1


eclobliquity = numpy.pi*(23. + (26 + 21.406/60.)/60.)/180.

def equecl(r, d):
    from numpy import sin, cos
    e = eclobliquity
    uv = lb2uv(r, d)
    uve = uv * 0
    uve[:,0] = uv[:,0]
    uve[:,1] = uv[:,1]*cos(e)+uv[:,2]*sin(e)
    uve[:,2] = -uv[:,1]*sin(e)+uv[:,2]*cos(e)
    return uv2lb(uve)

def eclequ(lam, be):
    from numpy import sin, cos
    e = eclobliquity
    uv = lb2uv(lam, be)
    uve = uv * 0
    uve[:,0] = uv[:,0]
    uve[:,1] = uv[:,1]*cos(e)-uv[:,2]*sin(e)
    uve[:,2] = uv[:,1]*sin(e)+uv[:,2]*cos(e)
    return uv2lb(uve)


# stolen from internet
def HMS2deg(ra='', dec='', delimiter=None):
  RA, DEC, rs, ds = '', '', 1, 1
  if dec:
    D, M, S = [float(i) for i in dec.split(delimiter)]
    if str(D)[0] == '-':
      ds, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    DEC = deg*ds
  
  if ra:
    H, M, S = [float(i) for i in ra.split(delimiter)]
    if str(H)[0] == '-':
      rs, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    RA = deg*rs
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

# stolen from internet
def deg2HMS(ra='', dec='', round=False, delimiter=' ', includedecsign=False,
            radecimals=4, decdecimals=3):
  RA, DEC, rs, ds = '', '', '', ''
  rawidth = radecimals+1+2
  decwidth = decdecimals+1+2
  if includedecsign:
      ds = '+'
  if dec:
    if str(dec)[0] == '-':
      ds, dec = '-', abs(dec)
    deg = int(dec)
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((dec-deg)*60)-decM)*60)
    else:
      decS = (abs((dec-deg)*60)-decM)*60
    DEC = '{0}{1:02}{d}{2:02}{d}{3:0{4}.{5}f}'
    DEC = DEC.format(ds, deg, decM, decS, decwidth, decdecimals, d=delimiter)
  
  if ra:
    if str(ra)[0] == '-':
      rs, ra = '-', abs(ra)
    raH = int(ra/15)
    raM = int(((ra/15)-raH)*60)
    if round:
      raS = int(((((ra/15)-raH)*60)-raM)*60)
    else:
      raS = ((((ra/15)-raH)*60)-raM)*60
    RA = '{0}{1:02}{d}{2:02}{d}{3:0{4}.{5}f}'
    RA = RA.format(rs, raH, raM, raS, rawidth, radecimals, d=delimiter)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

def map_coordinates(grid, coord):
    gridpts = grid[0]
    gridcoord = grid[1]
    outputcoordnorm = [numpy.interp(c, gc, numpy.arange(len(gc))) for c, gc in zip(coord, gridcoord)]
    from scipy.ndimage import interpolation
    return interpolation.map_coordinates(gridpts, outputcoordnorm,
                                         cval=0., mode='constant', order=1)

def hsv_to_rgb(hsv, rotate=0.5):
    """Convert HSV to RGB.

    Wrapper for matplotlib.colors.hsv_to_rgb, with the following change:
    if saturation is negative, rotate hue by rotate.  We also tune down
    the value of pixels with low saturation, so that in black and white equal
    value pixels will be roughly the same shade of gray.

    Args:
        hsv (ndarray[*,3]): hsv color array
        rotate (float): amount to rotate negative saturation by

    Returns:
        ndarray[*,3] rgb color array
    """
    from matplotlib import colors
    hsv = hsv.copy()
    m = hsv[...,1] < 0.
    hsv[m,0] = (hsv[m,0] + rotate) % 1.
    hsv[m,1] = -hsv[m,1]
    hsv[...,2] = (hsv[...,2] * (0.5 + hsv[...,1]/2.))
    return colors.hsv_to_rgb(hsv)


def mjd2utc(mjd):
    from astropy.time import Time
    t = Time(mjd, format='mjd')
    return t.isot


def utc2mjd(utc):
    from astropy.time import Time
    t = Time(utc, format='isot', scale='utc')
    return t.mjd


def repeated_linear(aa, bb, cinv, guess=None, damp=3):
    """Fit linear function with pseudo-Huber loss function via repeated SVD."""
    
    # Better to do this by giving a Jacobian to scipy.optimze.least_squares
    from scipy import sparse
    aa = numpy.sqrt(cinv).dot(aa)
    bb = numpy.sqrt(cinv).dot(bb)

    def chi(x):
        chi0 = damper(bb-aa.dot(x), damp)
        return chi0

    def jacobian(x):
        dd = damper_deriv(bb - aa.dot(x), damp)
        dd = sparse.diags(dd, 0)
        return -dd.dot(aa)

    if guess is None:
        guess = numpy.zeros(aa.shape[1])

    from scipy.optimize import least_squares
    res = least_squares(chi, guess, jac=jacobian)
    atcinva = res['jac'].T.dot(res['jac'])
    u, s, vh = numpy.linalg.svd(numpy.array(atcinva.todense()))
    s2 = s.copy()
    svdthresh = 1e-10
    s2[s < svdthresh] = 0.
    var, _ = svd_variance(u, s2, vh, no_covar=True)
    return res['x'].copy(), var, res


"""
Old problematic code.
    import scipy
    if guess is None:
        guess = numpy.zeros(aa.shape[1])
    aa = numpy.sqrt(cinv).dot(aa)
    bb = numpy.sqrt(cinv).dot(bb)
    for i in range(100):
        resid = bb-aa.dot(guess)
        dresid = damper(resid, damp=damp)
        d1 = damper_deriv(resid, damp=damp)
        d1 = scipy.sparse.diags(d1, 0)
        d2 = damper_deriv(resid, damp=damp, derivnum=2)*dresid
        d2 = scipy.sparse.diags(d2, 0)
        atcinva = aa.T.dot((d1**2-d2).dot(aa))
        atcinvb = aa.T.dot(d1.dot(dresid))
        step = scipy.sparse.linalg.cg(atcinva, atcinvb, tol=1e-9)[0]
        guess += step
        print(guess[-1], numpy.max(numpy.abs(step)))
        pdb.set_trace()
        if numpy.allclose(step, 0):
            break
        if i == 99:
            print('Not enough iterations!')
    u, s, vh = numpy.linalg.svd(numpy.array(atcinva.todense()))
    s2 = s.copy()
    svdthresh = 1e-10
    s2[s < svdthresh] = 0.
    step = svsol(u, s2, vh, atcinvb)
    var, _ = svd_variance(u, s2, vh, no_covar=True)
    return guess, var
"""


def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*numpy.sign(chi)*(numpy.sqrt(1+numpy.abs(chi)/damp)-1)
    # return chi/numpy.sqrt(1+numpy.abs(chi)/damp)


def damper_deriv(chi, damp, derivnum=1):
    """Derivative of the pseudo-Huber loss function."""
    if derivnum == 1:
        return (1+numpy.abs(chi)/damp)**(-0.5)
    if derivnum == 2:
        return -0.5*numpy.sign(chi)/damp*(1+numpy.abs(chi)/damp)**(-1.5)


def merge_arrays(arrlist):
    """takes a list of arrays, merges them by field, should be what
    np.lib.recfunctions.merge_arrays does, but maybe that does something
    much slower?"""

    sz = sum([len(arr) for arr in arrlist])
    out = numpy.zeros(sz, dtype=arrlist[0].dtype)
    count = 0
    for arr in arrlist:
        for name in arr.dtype.names:
            out[name][count:count+len(arr)] = arr[name]
        count += len(arr)
    return out
