import os
import random
import pdb
import fnmatch
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot
from util_efs_c import max_bygroup, add_arr_at_ind


# stolen from Mario Juric
def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)

    return np.degrees(
        2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 +
                       cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));


def sample(obj, n):
    ind = random.sample(range(len(obj)),np.int(n))
    return obj[np.array(ind)]


def random_pts_on_sphere(n, mask=None):
    import healpy
    xyz = np.random.randn(n, 3)
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
    dub = 2*np.sin(np.radians(rad)/2)
    if nneighbor > 0:
        d12, m2 = tree.query(uv1, nneighbor, distance_upper_bound=dub)
        if nneighbor > 1:
            m2 = m2.reshape(-1)
            d12 = d12.reshape(-1)

        m1 = np.arange(len(r1)*nneighbor, dtype='i4') // nneighbor
        d12 = 2*np.arcsin(np.clip(d12 / 2, 0, 1))*180/np.pi
        m = m2 < len(r2)
    else:
        tree1 = cKDTree(uv1)
        res = tree.query_ball_tree(tree1, dub)
        lens = [len(r) for r in res]
        m2 = np.repeat(np.arange(len(r2), dtype='i4'), lens)
        if len(m2) > 0:
            m1 = np.concatenate([r for r in res if len(r) > 0])
        else:
            m1 = m2.copy()
        d12 = gc_dist(r1[m1], d1[m1], r2[m2], d2[m2])
        m = np.ones(len(m1), dtype='bool')
    if notself:
        m = m & (m1 != m2)
    return m1[m], m2[m], d12[m]


def match2d_nearest(x1, y1, x2, y2, rad):
    """Find nearest match between x1, y1 and x2, y2 within radius rad."""
    from scipy.spatial import cKDTree
    xx1 = np.stack([x1, y1], axis=1)
    xx2 = np.stack([x2, y2], axis=1)
    tree1 = cKDTree(xx1)
    dd, ii = tree1.query(xx2, distance_upper_bound=rad)
    m = np.isfinite(dd)
    return ii[m], np.flatnonzero(m), dd[m]


def match2d(x1, y1, x2, y2, rad, nearest=False):
    """Find all matches between x1, y1 and x2, y2 within radius rad."""
    from scipy.spatial import cKDTree
    xx1 = np.stack([x1, y1], axis=1)
    xx2 = np.stack([x2, y2], axis=1)
    tree1 = cKDTree(xx1)
    tree2 = cKDTree(xx2)
    res = tree1.query_ball_tree(tree2, rad)
    lens = [len(r) for r in res]
    m1 = np.repeat(np.arange(len(x1), dtype='i4'), lens)
    if sum([len(r) for r in res]) == 0:
        m2 = m1.copy()
    else:
        m2 = np.concatenate([r for r in res if len(r) > 0])
    d12 = np.sqrt(np.sum((xx1[m1, :]-xx2[m2, :])**2, axis=1))
    if nearest:
        keep = np.zeros(len(m1), dtype='bool')
        for f, l in subslices(m1):
            keepind = np.argmin(d12[f:l])
            keep[f+keepind] = True
        m1 = m1[keep]
        m2 = m2[keep]
        d12 = d12[keep]
    return m1, m2, d12


def unique(obj):
    """Gives an array indexing the unique elements in the sorted list obj.

    Returns the list of indices of the last elements in each group
    of equal-comparing items in obj
    """
    nobj = len(obj)
    if nobj == 0:
        return np.zeros(0, dtype='i8')
    if nobj == 1:
        return np.zeros(1, dtype='i8')
    out = np.zeros(nobj, dtype=np.bool)
    out[0:nobj-1] = (obj[0:nobj-1] != obj[1:nobj])
    out[nobj-1] = True
    return np.sort(np.flatnonzero(out))


def unique_multikey(obj, keys):
    """Gives an array indexing the unique elements in the sorted list obj,
    according to a set of keys.

    Returns the list of indices of the last elements in each group
    of equal-comparing items in obj
    """
    nobj = len(obj[keys[0]])
    if nobj == 0:
        return np.zeros(0, dtype='i8')
    if nobj == 1:
        return np.zeros(1, dtype='i8')
    out = np.zeros(nobj, dtype=np.bool)
    for k in keys:
        out[0:nobj-1] |= (obj[k][0:nobj-1] != obj[k][1:nobj])
    out[nobj-1] = True
    return np.sort(np.flatnonzero(out))


def iqr(dat):
    from scipy.stats.mstats import mquantiles
    quant = mquantiles(dat, (0.25, 0.75))
    return quant[1]-quant[0]


def minmax(v, nan=False):
    v = np.asarray(v)
    if nan:
        return np.asarray([np.nanmin(v), np.nanmax(v)])
    return np.asarray([np.min(v), np.max(v)])


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


def svsol(u,s,vh,b): # N^2 time
    out = np.dot(np.transpose(u), b)
    s2 = 1./(s + (s == 0))*(s != 0)
    out = np.dot(np.diag(s2), out)
    out = np.dot(np.transpose(vh), out)
    return out

def svd_variance(u, s, vh, no_covar=False):
    s2 = 1./(s + (s == 0))*(s != 0)
#    covar = np.dot(np.dot(np.transpose(vh), np.diag(s2)),
#                      np.transpose(u))
    if no_covar: # computing the covariance matrix is expensive, n^3 time.  if we skip that, only n^2
        return (np.array([ np.sum(vh.T[i,:]*s2*u.T[:,i]) for i in range(len(s2))]),
                np.nan)
    covar = vh.T*s2.reshape(1,-1)
    covar = np.dot(covar, u.T)
    var = np.diag(covar)
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

    cs = np.cumsum(v, axis=0, dtype='f8')
    c = np.zeros((v.shape[0]+1, v.shape[1]))
    c[1:, :] = cs / (cs[-1, :] + (cs[-1, :] == 0))

    # Now find the desired
    #x   = np.arange(v.shape[0]+1) - 0.5
    x   = np.linspace(0, v.shape[0], v.shape[0]+1)
    res = np.empty(np.array(quantiles, copy=False).shape +
                      (v.shape[1],), dtype=float)
    norm = x*1e-10
    for k in range(v.shape[1]):
        # Construct interpolation object
        y  = c[:, k] + norm
        # this tiny fudge ensures y is always monotonically increasing
        res[:, k] = np.interp(quantiles, y, x)

    if rescale:
        if catchallbin:
            res = rescale[0] + (rescale[1]-rescale[0])*(res-1)/(v.shape[0]-2)
        else:
            res = rescale[0] + (rescale[1] - rescale[0]) * (res/v.shape[0])

    if axis != 0:
        v = v.transpose()

    res[:, cs[-1, :] == 0] = np.nan

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
    v = (np.atleast_1d(np.asarray(v))).flat
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan

    if len(v) == 1:
        return v[0], 0*v[0]

    from scipy.stats.mstats import mquantiles
    (ql, med, qu) = mquantiles(v, (0.25, 0.5, 0.75))
    iqr = qu - ql

    vmin = med - iqr * clip / 1.349
    vmax = med + iqr * clip / 1.349
    mask = (vmin <= v) & (v <= vmax) & np.isfinite(v)
    v = v[mask].astype(np.float64)
    if len(v) <= 0:
        pdb.set_trace()
    ret = np.mean(v), np.std(v, ddof=1)
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
    weight = weight if weight is not None else np.ones(len(x[0]),
                                                          dtype='u4')
    out = np.zeros(np.product(outdims), dtype=dtype)
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
        weight = np.ones(len(x[0]), dtype='u4')
    loc = [np.array(np.floor((i-f)//sz)+1, dtype='i4')
           for i,f,sz in zip(x, first, binsz)]
    loc = [np.clip(loc0, 0, n+1, out=loc0) for loc0, n in zip(loc, num)]
    flatloc = ravel_index(loc, outdims)
    return add_arr_at_ind(out, weight, flatloc)

def make_bins(p, range, bin, npix):
    if bin != None and npix != None:
        print("bin size and number of bins set; ignoring bin size")
        bin = None
    if range is not None and min(range) == max(range):
        raise ValueError('min(range) must not equal max(range)')
    flip = False
    m = np.isfinite(p)
    range = (range if range is not None else
             [np.min(p[m]), np.max(p[m])])
    if range[1] < range[0]:
        flip = True
        range = [range[1], range[0]]
    if bin != None:
        npix = np.ceil((range[1]-range[0])/bin)
        npix = npix if npix > 0 else 1
        pts = range[0] + np.arange(npix+1)*bin
        if bin == 0:
            pdb.set_trace()
    else:
        npix = (npix if npix is not None else
                np.ceil(0.3*np.sqrt(len(p))))
        npix = int(npix) if npix > 0 else 1
        pts = np.linspace(range[0], range[1], npix+1)
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
            weight = np.ones(len(x))
        m = np.isfinite(x) & np.isfinite(y)
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
        inf = np.array([np.inf])
        xpts2, ypts2 = [np.concatenate([-inf, i, inf])
                        for i in (xpts, ypts)]
        self.xe_a = xpts2.copy()
        self.ye_a = ypts2.copy()
        #self.hist_all, self.xe_a, self.ye_a = \
        #            np.histogram2d(x, y, bins=[xpts2, ypts2])
        self.deltx, self.delty = self.xe_a[2]-self.xe_a[1], self.ye_a[2]-self.ye_a[1]
        norm = abs(float(np.sum(self.hist_all)*self.deltx*self.delty)) if normalize else 1.
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
        zo = np.arange(2, dtype='i4')
        self.xrange = self.xpts[-zo]+0.5*self.deltx*(zo*2-1)
        self.yrange = self.ypts[-zo]+0.5*self.delty*(zo*2-1)
        self.coltot = np.sum(self.hist_all[1:-1,:], axis=1)
        # 1:-1 to remove the first and last rows in _x_ from the
        # conditional sums. e.g., everything that falls outside of
        # the x limits.  The stuff falling outside of the y limits
        # still counts
        if self.conditional:
            self.q16, self.q50, self.q84 = \
                    percentile_freq(self.hist_all[1:-1,:], [0.16, 0.5, 0.84],
                                    axis=1, rescale=[self.ye[0], self.ye[-1]])

    def show(self, linecolor='k', clipzero=False, steps=True, **kw):
        hist = self.hist
        if self.conditional:
            coltot = (self.coltot + (self.coltot == 0))
        else:
            coltot = np.ones(len(self.coltot))
        dispim = hist / coltot.reshape((len(coltot), 1))
        if clipzero:
            m = dispim == 0
            dispim[m] = np.min(dispim[~m])/2
        extent = [self.xe[0], self.xe[-1], self.ye[0], self.ye[-1]]
        if 'nograyscale' not in kw or not kw['nograyscale']:
            pyplot.imshow(dispim.T, extent=extent, interpolation='nearest',
                          origin='lower', aspect='auto', **kw)
        if self.conditional:
            xpts2 = (np.repeat(self.xpts, 2) +
                     np.tile(np.array([-1.,1.])*0.5*self.deltx,
                                len(self.xpts)))
            skip = 1 if steps else 2
            pyplot.plot(xpts2[::skip], np.repeat(self.q16, 2)[::skip],
                        linecolor,
                        xpts2[::skip], np.repeat(self.q50, 2)[::skip],
                        linecolor,
                        xpts2[::skip], np.repeat(self.q84, 2)[::skip],
                        linecolor)
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
        val = np.asarray(val)
        descr = val.dtype.descr
        if np.atleast_1d(val) is val:
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
            if np.product(val.shape) != 0:
                dtype += adddtype
            else:
                skipzero.append(f)
        elif val.dtype.fields is not None:
            dtype += [(f, val.dtype.descr)]
        else:
            dtype.append((f, val.dtype.str))
    out = np.zeros(1, dtype=dtype)
    for f in fields:
        if f in skipzero:
            print("field %s has zero size, skipping" % f)
            continue
        val = getfn(ob, f, None)
        if val is not None:
            if not np.isscalar(out[f][0]):
                val = np.array(val)
                assert out[f][0].shape == val.shape, (out[f][0].shape, val.shape)
                assert out[f][0].dtype == val.dtype, (out[f][0].dtype, val.dtype)
            out[f][0] = val
    return out

# np.void can't be pickled, so set novoid if you want to get an ndarray
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
    names = np.array([dt[0] for dt in newdtype])
    s = np.argsort(names)
    names = names[s]
    u = unique(names)
    nname = np.concatenate([[u[0]+1], u[1:]-u[0:-1]])
    if np.any(nname > 1):
        print('Duplicate names.', names[u[nname > 1]])
        raise ValueError('Duplicate column names in table.')
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
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
    s = np.argsort(val)
    if len(d) > 0:
        t = type(next(iter(d.values())))
    else:
        t = type(default)
    out = np.zeros(len(val), dtype=t)
    for first, last in subslices(val[s]):
        key = val[s[first]]
        if (not key in d) and (default is None):
            raise AttributeError('dictionary d missing key /%s/' % str(key))
        out[s[first:last]] = d.get(val[s[first]], default)
    return out

def mag_arr_index(mag, ind):
    return mag[:,ind] if isinstance(ind, int) else mag[ind]

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
    dat = np.atleast_1d(dat)
    if invvar is not None:
        invvar = np.atleast_1d(invvar)
        assert len(invvar) == len(dat)
        assert np.all(invvar >= 0)
    if removenan:
        keep = np.isfinite(dat)
        if invvar is not None:
            keep = keep & np.isfinite(invvar)
        dat = dat[keep]
        if invvar is not None:
            invvar = invvar[keep]
        initial_mask = keep
    nan = np.nan
    ngood = np.sum(invvar > 0) if invvar is not None else len(dat)
    if ngood == 0:
        ntot = len(dat)
        if removenan:
            ntot = len(initial_mask)
        out = {'mean':nan, 'median':nan, 'sigma':nan,
               'mask':np.zeros(ntot, dtype='bool'), 'newivar':nan}
        return out
    if ngood == 1:
        val = dat[invvar > 0] if invvar is not None else dat[0]
        out = {'mean':val, 'median':val, 'sigma':0.,
               'mask':np.ones(1, dtype='bool'), 'newivar':nan}
        if invvar is not None:
            out['newivar'] = invvar[invvar > 0][0]
        return out
    if invvar is not None:
        mask = invvar > 0
    else:
        mask = np.ones_like(dat, dtype='bool')
    if prefilter:
        w = invvar if invvar is not None else None
        quart = weighted_quantile(dat, weight=w, quant=[0.25,0.5, 0.75],
                                  interp=True)
        iqr = quart[2]-quart[0]
        med = quart[1]
        mask = mask & (np.abs(dat-med) <= sigrej*iqr)
    if invvar is not None:
        invsig = np.sqrt(invvar)
        fmean = np.sum(dat*invvar*mask)/np.sum(invvar*mask)
    else:
        fmean = np.sum(dat*mask)/np.sum(mask)
    fsig = np.sqrt(np.sum((dat-fmean)**2.*mask)/(ngood-1))
    iiter = 1
    savemask = mask

    nlast = -1
    while iiter < maxiter and nlast != ngood and ngood >= 2:
        nlast = ngood
        iiter += 1
        if invvar is not None:
            mask = (np.abs(dat-fmean)*invsig < sigrej) & (invvar > 0)
        else:
            mask = np.abs(dat-fmean) < sigrej*fsig
        ngood = np.sum(mask)
        if ngood >= 2:
            if invvar is not None:
                fmean = np.sum(dat*invvar*mask) / np.sum(invvar*mask)
            else:
                fmean = np.sum(dat*mask) / ngood
            fsig = np.sqrt(np.sum((dat-fmean)**2.*mask) / (ngood-1))
            savemask = mask
    fmedian = np.median(dat[savemask])
    newivar = nan
    if invvar is not None:
        newivar = np.sum(invvar*savemask)
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
    return np.all(x[1:len(x)] >= x[0:-1])

def solve_lstsq(aa, bb, ivar, svdthresh=None, return_covar=False):
    d, t = np.dot, np.transpose
    atcinvb = d(t(aa), ivar*bb)
    atcinva = d(t(aa), ivar.reshape((len(ivar), 1))*aa)
    u,s,vh = np.linalg.svd(atcinva)
    if svdthresh is not None:
        s[s < svdthresh] = 0.
    par = svsol(u,s,vh,atcinvb)
    var, covar = svd_variance(u, s, vh)
    ret = (par, var)
    if return_covar:
        ret = ret + (covar,)
    return ret

def solve_lstsq_covar(aa, bb, icvar, svdthresh=None, return_covar=False):
    d, t = np.dot, np.transpose
    atcinvb = d(t(aa), d(icvar,bb))
    atcinva = d(t(aa), d(icvar,aa))
    u,s,vh = np.linalg.svd(atcinva)
    if svdthresh is not None:
        s[s < svdthresh] = 0.
    par = svsol(u,s,vh,atcinvb)
    var, covar = svd_variance(u, s, vh)
    ret = (par, var)
    if return_covar:
        ret = ret + (covar,)
    return ret

def polyfit(x, y, deg, ivar=None, return_covar=False):
    aa = np.zeros((len(x), deg+1))
    for i in range(deg+1):
        aa[:,i] = x**(deg-i)
    if ivar is None:
        ivar = np.ones_like(y)
    m = np.isfinite(ivar)
    m[m] &= ivar[m] > 0
    ivar = ivar.copy()
    ivar[~m] = 0
    return solve_lstsq(aa, y, ivar, return_covar=return_covar)

def poly_iter(x, y, deg, yerr=None, sigrej=3., niter=10, return_covar=False,
              return_mask=False):
    m = np.ones_like(x, dtype='bool')
    if yerr is None:
        invvar = None
    else:
        invvar = 1./yerr**2
    for i in range(niter):
        iv = invvar[m] if invvar is not None else None
        sol = polyfit(x[m], y[m], deg, ivar=iv, return_covar=return_covar)
        p = sol[0]
        if np.sum(m) <= 1:
            break
        res = y - np.polyval(p, x)
        if invvar is None:
            stats = djs_iterstat(res[m], invvar=iv, sigrej=sigrej,
                                 prefilter=True)
            mn, sd = clipped_stats(res[m], clip=sigrej)
            m2 = np.abs(res - mn)/(sd+(sd == 0)) < sigrej
        else:
            chi = res*invvar**0.5
            stats = djs_iterstat(chi[m], prefilter=True)
            mn, sd = clipped_stats(chi[m], clip=sigrej)
            m2 = np.abs(chi) < sigrej
        if np.all(m2 == m):
            break
        else:
            m = m2
    if return_mask:
        sol = sol + (m,)
    return sol

def weighted_quantile(x, weight=None, quant=[0.25, 0.5, 0.75], interp=False):
    if weight is None:
        weight = np.ones_like(x)
    weight = weight / np.float(np.sum(weight))
    s = np.argsort(x)
    weight = weight[s]
    if interp:
        weight1 = np.cumsum(weight)
        weight2 = 1.-(np.cumsum(weight[::-1])[::-1])
        weight = (weight1 + weight2)/2.
        pos = np.interp(quant, weight, np.arange(len(weight)))
        quant = np.interp(pos, np.arange(len(weight)), x[s])
    else:
        weight = np.cumsum(weight)
        weight /= weight[-1]
        pos = np.searchsorted(weight, quant)
        quant = np.zeros(len(quant))
        quant[pos <= 0] = x[s[0]]
        quant[pos >= len(x)] = x[s[-1]]
        m = (pos > 0) & (pos < len(x))
        quant[m] = x[s[pos[m]]]
    return quant

def lb2tp(l, b):
    return (90.-b)*np.pi/180., l*np.pi/180.

def tp2lb(t, p):
    return p*180./np.pi % 360., 90.-t*180./np.pi

def tp2uv(t, p):
    z = np.cos(t)
    x = np.cos(p)*np.sin(t)
    y = np.sin(p)*np.sin(t)
    return np.concatenate([q[...,np.newaxis] for q in (x, y, z)],
                             axis=-1)
    #return np.vstack([x, y, z]).transpose().copy()

def lb2uv(r, d):
    return tp2uv(*lb2tp(r, d))

def uv2tp(uv):
    norm = np.sqrt(np.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = np.arccos(uv[:,2])
    p = np.arctan2(uv[:,1], uv[:,0])
    return t, p

def xyz2tp(x, y, z):
    norm = np.sqrt(x**2+y**2+z**2)
    t = np.arccos(z/norm)
    p = np.arctan2(y/norm, x/norm)
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
    l, b = np.radians(l), np.radians(b)
    z = re * np.sin(b)
    x = r0 - re*np.cos(l)*np.cos(b)
    y = -re*np.sin(l)*np.cos(b)
    return x, y, z

def xyz_galactic2lbr(x, y, z, r0=8.5):
    """See note about this coordinate system in lbr2xyz_galactic."""
    xe = r0-x
    re = np.sqrt(xe**2+y**2+z**2)
    b = np.degrees(np.arcsin(z / re))
    l = np.degrees(np.arctan2(-y, xe)) % 360.
    return l, b, re


def xyz2rphiz(x, y, z):
    r = np.sqrt(x**2+y**2)
    phi = np.degrees(np.arctan2(y, x))
    return r, phi, z


# should write a lbr2uvw for the right handed coordinate system.
# galpy apparently uses the left handed coordinate system, though.

def lbr2uvw_galactic(l, b, re):
    """Right handed, U increasing toward the GC, V increasing toward l=90,
    W increasing toward the NGC.  Origin at the earth."""

    l, b = np.radians(l), np.radians(b)
    w = re*np.sin(b)
    u = re*np.cos(l)*np.cos(b)
    v = re*np.sin(l)*np.cos(b)
    # very familiar!
    return u, v, w


def uvw_galactic2lbr(u, v, w):
    re = np.sqrt(u**2+v**2+w**2)
    b = np.degrees(np.arcsin(w/re))
    l = np.degrees(np.arctan2(v, u)) % 360.
    return l, b, re


def healgen(nside):
    import healpy
    return healpy.pix2ang(nside, np.arange(12*nside**2))

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
    newmap = np.sum(newmap, axis=1)
    if ring:
        newmap = healpy.reorder(newmap, n2r=True)
    return newmap / binfac

def heal_rebin_mask(map, nside, mask, ring=True, nanbad=False):
    newmap = heal_rebin(map*mask, nside, ring=ring)
    newmask = heal_rebin(mask, nside, ring=ring)
    out = newmap/(newmask + (newmask == 0))
    if nanbad:
        out[newmask == 0] = np.nan
    return out

def heal2cart(heal, interp=True, return_pts=False):
    import healpy
    nside = healpy.get_nside(heal)#*(2 if interp else 1)
    owidth = 8*nside
    oheight = 4*nside-1
    dm,rm = np.mgrid[0:oheight,0:owidth]
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
        map = (map, np.sort(np.unique(rm)), np.sort(np.unique(dm)))
    return map


def imshow(im, xpts=None, ypts=None, xrange=None, yrange=None, range=None,
           min=None, max=None, mask_nan=False,
           interp_healpy=True, log=False, center_gal=False, center_l=None, color=False,
           return_handler=False, contour=None,
           **kwargs):
    if xpts is not None and ypts is not None:
        dx = np.median(xpts[1:]-xpts[:-1])
        dy = np.median(ypts[1:]-ypts[:-1])
        kwargs['extent'] = [xpts[0]-dx/2., xpts[-1]+dx/2.,
                            ypts[0]-dy/2., ypts[-1]+dy/2.]
        if len(xpts) != im.shape[0] or len(ypts) != im.shape[1]:
            print('Warning: mismatch between xpts, ypts and im.shape')
        if not color:
            im = im.T
        else:
            im = np.transpose(im, axes=[1, 0, 2])
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
            for i in np.arange(ncolor, dtype='i4'):
                oneim = heal2cart(im[:,i], interp=interp_healpy)
                if outim is None:
                    outim = np.zeros(oneim.shape+(ncolor,))
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
                im = np.roll(im, np.int((im.shape[1]*left_l)/360), axis=1)
            else:
                ncolor = im.shape[-1]
                for i in np.arange(ncolor,dtype='i4'):
                    im[:,:,i] = np.roll(im[:,:,i], np.int(im.shape[1]*left_l/360), axis=1)
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
    if mask_nan and np.all(kwargs['cmap'].get_bad() == 0):
        #im = np.ma.array(im, mask=np.isnan(im))
        kwargs['cmap'].set_bad('lightblue', 0.1)
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
        self.min, self.max = (np.nanmin(np.array(self.im1._A)),
                              np.nanmax(np.array(self.im1._A)))
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
            self.min, self.max = (np.nanmin(np.array(self.im1._A)),
                                  np.nanmax(np.array(self.im1._A)))
        if isinstance(self.im1.norm, matplotlib.colors.LogNorm):
            if self.min <= 0:
                self.min = 1 # dumb
            self.min, self.max = np.log10([self.min, self.max])
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
        maximage = np.float(np.max(sc.hist*sc.norm))
        if log:
            if logspace is None:
                levels = 10.**(logmin + (np.arange(nlevels)+1)*
                               (np.log10(maximage)-logmin)/nlevels)/sc.norm
            else:
                levels = 10.**(np.log10(maximage)-logspace*np.arange(nlevels))[::-1]/sc.norm
        else:
            levels = (np.arange(nlevels)+minlevel)*maximage/sc.norm/nlevels
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
    xloc = np.array(np.floor((x-sc.xe[0]) #[0 if not flipx else -1])
                                   /sc.deltx), dtype='i4')
    yloc = np.array(np.floor((y-sc.ye[0]) # if not flipy else -1])
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
    sa = np.argsort(a)
    sb = np.argsort(b)
    ua = unique(a[sa])
    ub = unique(b[sb])
    if len(ua) != len(a):# or len(ub) != len(b):
        raise ValueError('All keys in a must be unique.')
    ind = np.searchsorted(a[sa], b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], np.flatnonzero(m)

def match_sorted_unique(a, b):
    ind = np.searchsorted(a, b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[ind[m]] == b[m]
    m[m] &= matches
    return ind[m], np.flatnonzero(m)

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
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)]
    return eval(''.join(evList))

def data2map(data, l, b, weight=None, nside=512, finiteonly=True):
    if weight is None:
        weight = np.ones(len(data))
    if finiteonly:
        m = (np.isfinite(data) & np.isfinite(weight) &
             np.isfinite(l) & np.isfinite(b))
        data = data[m]
        l = l[m]
        b = b[m]
        weight = weight[m]
    t, p = lb2tp(l, b)
    import healpy
    pix = healpy.ang2pix(nside, t, p)
    out = np.zeros(12*nside**2)
    wmap = np.zeros_like(out)
    out = add_arr_at_ind(out, weight*data, pix)
    wmap = add_arr_at_ind(wmap, weight, pix)
    out = out / (wmap + (wmap == 0))
    return out, wmap


def paint_map(r, d, dat, rad, weight=None, nside=512):
    import healpy
    npix = 12*nside**2
    vec = healpy.ang2vec(*lb2tp(r, d))
    map = np.zeros(npix)
    wmap = np.zeros(npix)
    if weight is None:
        weight = np.ones(len(dat), dtype='i4')
    for v, d, w in zip(vec, dat, weight):
        pix = healpy.query_disc(nside, v, rad*np.pi/180.)
        map[pix] += d
        wmap[pix] += w
    map = map / (wmap + (wmap == 0))
    return map, wmap


def bindatan(coord, dat, weight=None, npix=None, ranges=None, bins=None):
    m = np.logical_and.reduce([np.isfinite(c) for c in coord])
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
    bins = [np.median(tpts[1:]-tpts[0:-1]) for tpts in pts]
    ranges = ranges0

    if weight is None:
        weight = np.ones(len(dat))
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
            hist_all = np.flip(hist_all, i)
            whist_all = np.flip(whist_all, i)
            ranges[i] = [r for r in reversed(ranges[i])]
    ptscen = [(tpts[:-1]+tpts[1:])/2. for tpts in pts]
    return (hist_all/(whist_all + (whist_all == 0)), whist_all.copy(),
            ptscen)


def bindata(x, y, dat, weight=None, xnpix=None, ynpix=None, xrange=None,
            yrange=None, xbin=None, ybin=None):
    # could replace with call to bindatan
    m = np.isfinite(x) & np.isfinite(y)
    xrange, xpts, flipx = make_bins(x[m], xrange, xbin, xnpix)
    yrange, ypts, flipy = make_bins(y[m], yrange, ybin, ynpix)
    xbin = np.median(xpts[1:]-xpts[0:-1])
    ybin = np.median(ypts[1:]-ypts[0:-1])

    if weight is None:
        weight = np.ones(len(dat))
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
    m = np.isfinite(x) & np.isfinite(y)
    xrange, xpts, flipx = make_bins(x[m], xrange, xbin, xnpix)
    yrange, ypts, flipy = make_bins(y[m], yrange, ybin, ynpix)
    xbin = np.median(xpts[1:]-xpts[0:-1])
    ybin = np.median(ypts[1:]-ypts[0:-1])
    if weight is None:
        weight = np.ones(len(dat))
    m = np.isfinite(dat)
    dat = dat[m]
    x = x[m]
    y = y[m]
    weight = weight[m]
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
                          weight=np.ones(len(dat)))

    # uhh, dumbest thing is max one point per bin
    m = count_hist <= pointcut
    hist_all = hist_all / (whist_all + (whist_all == 0.))
    hist_all[m] = np.nan
    if vmin is None:
        vmin = np.nanmin(hist_all)
    if vmax is None:
        vmax = np.nanmax(hist_all)

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


    loc = [np.array(np.floor((i-f)//sz)+1, dtype='i4')
           for i,f,sz in zip((x, y), (xpts[0], ypts[0]), (xpts[1]-xpts[0], ypts[1]-ypts[0]))]
    loc = tuple([np.clip(loc0, 0, n+1, out=loc0) for loc0, n in
                 zip(loc, (len(xpts)-1, len(ypts)-1))])

    xcen = (xpts[:-1] + xpts[1:])/2
    ycen = (ypts[:-1] + ypts[1:])/2
    ret = imshow(hist_all[1:-1,1:-1], xcen, ycen, xrange=xrange, yrange=yrange, cmap=cmap, log=log, vmin=vmin, vmax=vmax, origin='lower', mask_nan=mask_nan)
    if showpts:
        # need to find the points that were in a bin with only 1 point.
        m = ~np.isfinite(hist_all[loc])
        ret = pyplot.scatter(x[m], y[m], c=dat[m], edgecolor='none', vmin=vmin, vmax=vmax, cmap=cmap, **kw)
    if contour:
        if levels is None:
            maximage = np.float(np.max(count_hist[1:-1,1:-1]))
            if logcontour:
                if logspace is None:
                    levels = 10.**(logmin + (np.arange(nlevels)+1)*
                                   (np.log10(maximage)-logmin)/nlevels)
                else:
                    levels = 10.**(np.log10(maximage)-logspace*np.arange(nlevels))
            else:
                levels = (np.arange(nlevels)+minlevel)*maximage/nlevels

        pyplot.contour(count_hist[1:-1,1:-1].T,
                       extent=[xpts[0], xpts[-1], ypts[0], ypts[-1]],
                       colors=colors, levels=levels)
    return ret

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
    newnd = np.zeros(len(nd), dtype=newdtype)
    for col in nd.dtype.names:
        newnd[col] = nd[col]
    return newnd


def fitstondarray(fits, lower=False):
    from copy import deepcopy
    nd = np.zeros(len(fits), fits.dtype)
    for f in fits.dtype.names:
        nd[f] = fits[f]
    if lower:
        names = list(deepcopy(nd.dtype.names))
        names = [n.lower() for n in names]
        nd.dtype.names = names
    return nd


def stirling_approx(n):
    return n*np.log(n) - n + np.log(n)/2. + np.log(2*np.pi)/2.


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


def zenithrd(lat, lng, mjd, precess=True):
    lst = mjd2lst(mjd, lng)
    rr = lst*360./24
    dd = lat*np.ones_like(rr)
    if precess:
        import precess as precessmod
        jd2000   = 2451545.0
        mjdstart = 2400000.5
        rr, dd = precessmod.precess(
            rr, dd, 2000.+(mjd+mjdstart-jd2000)/365.25, 2000.)
    return rr, dd


def parallactic_angle(rr, dd, lat, lng, mjd):  # no precession
    ha = mjd2lst(mjd, lng)*360/24 - rr
    latr = np.radians(lat)
    har = np.radians(ha)
    ddr = np.radians(dd)
    from numpy import sin, cos
    parallactic = -np.degrees(np.arctan2(
        -np.sin(har),
        np.cos(ddr)*np.tan(latr)-np.sin(ddr)*np.cos(har)))
    return parallactic


# KPNO lat/lon: 31.960595 -111.599208
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
    d2r = np.pi/180.
    sin, cos = np.sin, np.cos
    sh, ch = sin(ha*d2r),  cos(ha*d2r)
    sd, cd = sin(dec*d2r), cos(dec*d2r)
    sl, cl = sin(lat*d2r), cos(lat*d2r)
    x = - ch * cd * sl + sd * cl
    y = - sh * cd
    z = ch * cd * cl + sd * sl
    r = np.sqrt(x**2 + y**2)
    az = (np.arctan2(y, x) / d2r) % 360.
    alt = (np.arctan2(z, r) / d2r)
    return alt, az


def altaz2hadec(alt, az, lat):
    d2r = np.pi/180.
    # altaz2hadec.pro in idlutils
    sin, cos = np.sin, np.cos
    salt, calt = sin(alt*d2r),  cos(alt*d2r)
    saz, caz = sin(az*d2r), cos(az*d2r)
    sl, cl = sin(lat*d2r), cos(lat*d2r)
    x = - caz * calt * sl + salt * cl
    y = - saz * calt
    z = caz * calt * cl + salt * sl
    r = np.sqrt(x**2 + y**2)
    ha = np.arctan2(y, x) / d2r
    d = (np.arctan2(z, r) / d2r) % 360.
    return ha, d


def altazllmjd2rd(alt, az, lat, lng, mjd, precess=True):
    d2r = np.pi/180.
    # altaz2hadec.pro in idlutils
    sin, cos = np.sin, np.cos
    salt, calt = sin(alt*d2r),  cos(alt*d2r)
    saz, caz = sin(az*d2r), cos(az*d2r)
    sl, cl = sin(lat*d2r), cos(lat*d2r)
    x = - caz * calt * sl + salt * cl
    y = - saz * calt
    z = caz * calt * cl + salt * sl
    r = np.sqrt(x**2 + y**2)
    ha = np.arctan2(y, x) / d2r
    d = (np.arctan2(z, r) / d2r) % 360.
    lst = mjd2lst(mjd, lng)
    r = lst*360./24 - ha
    if precess:
        # precess back to J2000 from observed coordinates.
        import precess as precessmod
        jd2000 = 2451545.0
        mjdstart = 2400000.5
        r, d = precessmod.precess(r, d, 2000.+(mjd + mjdstart - jd2000)/365.25, 2000.)
    return r, d


def alt2airmass(alt):
    # Pickering (2002) according to Wikipedia?
    #h = alt #90-alt
    #return 1./(np.sin(np.radians(h + 244 / (165 + 47*h**1.1))))
    # disagrees with sec(z) by 0.001 - 0.002 near zenith, which is the most important part anyway.
    # probably ~1% better at airmass 3, and rapidly better after that---but who cares?
    return 1./np.cos(np.radians(90-alt))


def refraction_simple_kpno(alt):
    return 220e-6*180/np.pi*np.tan(np.radians(90-alt))


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
    pix = wcs.wcs_sky2pix(np.array([r, d,
                                       np.ones(len(r)),
                                       np.ones(len(r))]).transpose(), 0)
    from scipy.ndimage import map_coordinates
    val = map_coordinates(ims, (pix[:,0:2])[:,::-1].transpose(),
                 cval=np.nan, order=1)
    return val


def write_file_in_chunks(dat, filename, chunksize, breakkey=None):
    from astropy.io import fits
    nbytes = len(dat)*dat.dtype.itemsize
    if breakkey is not None:
        s = np.argsort(dat[breakkey])
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


eclobliquity = np.pi*(23. + (26 + 21.406/60.)/60.)/180.

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
def deg2HMS(ra=None, dec=None, round=False, delimiter=' ', includedecsign=False,
            radecimals=4, decdecimals=3):
  RA, DEC, rs, ds = '', '', '', ''
  rawidth = radecimals+1+2
  decwidth = decdecimals+1+2
  if includedecsign:
      ds = '+'
  if dec is not None:
    if dec < 0:
      ds, dec = '-', abs(dec)
    deg = int(dec)
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((dec-deg)*60)-decM)*60)
    else:
      decS = (abs((dec-deg)*60)-decM)*60
    DEC = '{0}{1:02}{d}{2:02}{d}{3:0{4}.{5}f}'
    DEC = DEC.format(ds, deg, decM, decS, decwidth, decdecimals, d=delimiter)

  if ra is not None:
    if ra < 0:
      rs, ra = '-', abs(ra)
    raH = int(ra/15)
    raM = int(((ra/15)-raH)*60)
    if round:
      raS = int(((((ra/15)-raH)*60)-raM)*60)
    else:
      raS = ((((ra/15)-raH)*60)-raM)*60
    RA = '{0}{1:02}{d}{2:02}{d}{3:0{4}.{5}f}'
    RA = RA.format(rs, raH, raM, raS, rawidth, radecimals, d=delimiter)

  if ra is not None and dec is not None:
    return (RA, DEC)
  else:
    return RA if ra is not None else DEC


def map_coordinates(grid, coord):
    gridpts = grid[0]
    gridcoord = grid[1]
    outputcoordnorm = [np.interp(c, gc, np.arange(len(gc))) for c, gc in zip(coord, gridcoord)]
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
    aa = np.sqrt(cinv).dot(aa)
    bb = np.sqrt(cinv).dot(bb)

    def chi(x):
        chi0 = damper(bb-aa.dot(x), damp)
        return chi0

    def jacobian(x):
        dd = damper_deriv(bb - aa.dot(x), damp)
        dd = sparse.diags(dd, 0)
        return -dd.dot(aa)

    if guess is None:
        guess = np.zeros(aa.shape[1])

    from scipy.optimize import least_squares
    res = least_squares(chi, guess, jac=jacobian)
    atcinva = res['jac'].T.dot(res['jac'])
    if getattr(atcinva, 'todense', None) is not None:
        atcinva = atcinva.todense()
    u, s, vh = np.linalg.svd(np.array(atcinva))
    s2 = s.copy()
    svdthresh = 1e-10
    s2[s < svdthresh] = 0.
    var, _ = svd_variance(u, s2, vh, no_covar=True)
    return res['x'].copy(), var, res


def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*np.sign(chi)*(np.sqrt(1+np.abs(chi)/damp)-1)
    # return chi/np.sqrt(1+np.abs(chi)/damp)


def damper_deriv(chi, damp, derivnum=1):
    """Derivative of the pseudo-Huber loss function."""
    if derivnum == 1:
        return (1+np.abs(chi)/damp)**(-0.5)
    if derivnum == 2:
        return -0.5*np.sign(chi)/damp*(1+np.abs(chi)/damp)**(-1.5)


def merge_arrays(arrlist):
    """takes a list of arrays, merges them by field, should be what
    np.lib.recfunctions.merge_arrays does, but maybe that does something
    much slower?"""

    sz = sum([len(arr) for arr in arrlist])
    out = np.zeros(sz, dtype=arrlist[0].dtype)
    count = 0
    for arr in arrlist:
        for name in arr.dtype.names:
            out[name][count:count+len(arr)] = arr[name]
        count += len(arr)
    return out


def lb2tan(l, b, lcen=None, bcen=None):
    up = np.array([0, 0, 1])
    uv = lb2uv(l, b)
    if lcen is None:
        lcen, bcen = uv2lb(np.mean(uv, axis=0).reshape(1, -1))
        lcen, bcen = lcen[0], bcen[0]
    uvcen = lb2uv(lcen, bcen)
    # error if directly at pole
    rahat = np.cross(up, uvcen)
    rahat /= np.sqrt(np.sum(rahat**2))
    dechat = np.cross(uvcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2))
    xx = np.einsum('i,ji', rahat, uv)
    yy = np.einsum('i,ji', dechat, uv)
    xx *= 180/np.pi
    yy *= 180/np.pi
    return xx, yy


def tan2lb(xx, yy, lcen, bcen):
    uvcen = lb2uv(lcen, bcen)
    up = np.array([0, 0, 1])
    rahat = np.cross(up, uvcen)
    rahat /= np.sqrt(np.sum(rahat**2))
    dechat = np.cross(uvcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2))
    xcoord = xx*np.pi/180
    ycoord = yy*np.pi/180
    zcoord = np.sqrt(1-xcoord**2-ycoord**2)
    uv = (xcoord.reshape(-1, 1)*rahat.reshape(1, -1) +
          ycoord.reshape(-1, 1)*dechat.reshape(1, -1) +
          zcoord.reshape(-1, 1)*uvcen.reshape(1, -1))
    print(uvcen, rahat, dechat)
    return uv2lb(uv)


def compare_fits_files(fn1, fn2, hdunames=None):
    from astropy.io import fits
    hdul1 = fits.open(fn1)
    hdul2 = fits.open(fn2)
    if hdunames is None:
        hdunames1 = [hdu.name for hdu in hdul1]
        hdunames2 = [hdu.name for hdu in hdul2]
        if not np.all([a == b for a, b in zip(hdunames1, hdunames2)]):
            print('mismatching hdu names, failing')
            return False
        hdunames = [h for h in hdunames1 if h != 'PRIMARY']
    allequal = True
    for hduname in hdunames:
        dat1 = hdul1[hduname].data
        dat2 = hdul2[hduname].data
        for field in dat1.dtype.names:
            try:
                notequal = ~np.isclose(dat1[field], dat2[field],
                                       equal_nan=True)
            except TypeError:
                notequal = dat1[field] != dat2[field]
            if np.sum(notequal) != 0:
                print(hduname, field, np.sum(notequal))
            allequal &= (np.sum(notequal) == 0)
    return allequal



def foci_to_ellipse(foci, semimajor):
    dfocus = gc_dist(*np.concatenate([foci[0], foci[1]]))
    semiminor = np.sqrt(semimajor**2 - (dfocus/2)**2)
    uv = lb2uv(np.array(foci[:, 0]), np.array(foci[:, 1]))
    uvcen = np.sum(uv, axis=0)
    norm = np.sqrt(np.sum(uvcen**2))
    uvcen /= norm
    lcen, bcen = uv2lb(uvcen[None, :])
    lcen = lcen[0]
    bcen = bcen[0]
    # a, b, center, position angle...
    up = np.array([0, 0, 1])
    rahat = np.cross(up, uvcen)
    rahat /= np.sqrt(np.sum(rahat**2))
    dechat = np.cross(uvcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2))
    xx = np.dot(rahat, uv[0]-uvcen)
    yy = np.dot(dechat, uv[0]-uvcen)
    pa = 90 - np.degrees(np.arctan2(yy, xx))
    # arctan2 is from +RA toward +dec; we want from
    # +dec toward +RA.
    return semimajor, semiminor, lcen, bcen, pa
