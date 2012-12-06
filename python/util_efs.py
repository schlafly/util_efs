import random
import numpy
from scipy.stats.mstats import mquantiles
from lsd import DB
from lsd import bounds as lsdbounds
import os
import h5py
import matplotlib
from matplotlib import pyplot
import pdb
from scipy import weave
import cPickle as pickle
import tempfile
import pstats
from astrometry.libkd.spherematch import match_radec
from lsd.utils import gc_dist
from itertools import izip
import fnmatch
import healpy

def sample(obj, n):
    ind = random.sample(xrange(len(obj)),numpy.int(n))
    return obj[numpy.array(ind)]


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
    nobj = len(obj)
    if nobj == 0:
        return numpy.zeros(0, dtype='i8')
    if nobj == 1:
        return numpy.zeros(1, dtype='i8')
    out = numpy.zeros(nobj, dtype=numpy.bool)
    for k in keys:
        out[0:nobj-1] |= (obj[0:nobj-1][k] != obj[1:nobj][k])
    out[nobj-1] = True
    return numpy.sort(numpy.flatnonzero(out))

def max_bygroup(val, ind):
    out = numpy.zeros(len(ind), dtype=val.dtype)
    code = """
           #line 64 "util_efs.py"
           int pind = 0;
           for(int i=0; i<Nind[0]; i++) {
               int maxval = val(pind);
               for(int j=pind+1; j<ind(i)+1; j++) {
                   maxval = maxval > val(j) ? maxval : val(j);
               }
               out(i) = maxval;
               pind = ind(i)+1;
           }
           """
    if max(ind) > len(val):
        raise ValueError('Incompatible argument sizes.')
    if len(ind) == 0:
        return numpy.array([])
    if min(ind) < 0:
        raise ValueError('min(ind) must be >= 0')
    weave.inline(code, ['val', 'ind', 'out'],
                 type_converters=weave.converters.blitz, verbose=1)
    return out

def iqr(dat):
    quant = mquantiles(dat, (0.25, 0.75))
    return quant[1]-quant[0]

def minmax(v, nan=False):
    v = numpy.asarray(v)
    if nan:
        return numpy.asarray([numpy.nanmin(v), numpy.nanmax(v)])
    return numpy.asarray([numpy.min(v), numpy.max(v)])

class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data):
        self.uind = unique(data)
        self.ind = 0
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.uind)
    def next(self):
        if self.ind == len(self.uind):
            raise StopIteration
        if self.ind == 0:
            first = 0
        else:
            first = self.uind[self.ind-1]+1
        last = self.uind[self.ind]+1
        self.ind += 1
        return first, last
#        return slice(first, self.uind[self.ind]+1)


full_ps1_db = '/raid14/home/mjuric/lsd/full_ps1/lsd/db_20101203.1'
sas_db = '/raid14/home/mjuric/lsd/db_20101203.1'


def query_lsd(querystr, db=None, bounds=None, **kw):
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


def plothist_efs(dat, binsz=None, range=None):
    if binsz is None:
        binsz = 1.
    if range is None:
        range = [numpy.min(dat), numpy.max(dat)]
    nbin = float(range[1]-range[0])
#    return hist(


def svsol(u,s,vh,b):
    out = numpy.dot(numpy.transpose(u), b)
    s2 = 1./(s + (s == 0))*(s != 0)
    out = numpy.dot(numpy.diag(s2), out)
    out = numpy.dot(numpy.transpose(vh), out)
    return out

def svd_variance(u, s, vh):
    s2 = 1./(s + (s == 0))*(s != 0)
    covar = numpy.dot(numpy.dot(numpy.transpose(vh), numpy.diag(s2)),
                      numpy.transpose(u))
    var = numpy.diag(covar)
    return var, covar

def writehdf5(dat, filename, dsname=None, mode='a'):
    if dsname == None:
        dsname = 'default'
    f = h5py.File(filename, mode)
    try:
        f.create_dataset(dsname, data=dat)
        f.close()
    except Exception as e:
        f.close()
        raise e

def readhdf5(filename, dsname=None):
    f = h5py.File(filename, 'r')
    if dsname is None:
        keys = f.keys()
        if len(keys) == 0:
            f.close()
            raise IOError('No data found in file %s' % filename)
        dsname = keys[0]
    try:
        ds = f.create_dataset(dsname)
    except Exception as e:
        print 'possible keys:' , f.keys()
        f.close()
        raise e
    dat = ds[:]
    f.close()
    return dat

# stolen from internet, Simon Brunning
def locate(pattern, root=os.curdir):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
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
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
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
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
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
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None

# Stolen from MJ
#######################################################################
# Compute percentiles of 2D ndarrays of frequencies
def percentile_freq(v, quantiles, axis=0, rescale=None):
	"""
	Given an array v, compute quantiles along a given axis
	
	Array v is assumed to be two dimensional.
	"""
	if v.dtype != float:
		v = v.astype(float)
	if axis != 0:
		v = v.transpose()

	cs = numpy.cumsum(v, axis=0)
	c = numpy.zeros((v.shape[0]+1, v.shape[1]))
        c[1:, :] = cs / (cs[-1, :] + (cs[-1, :] == 0))

	# Now find the desired
	#x   = numpy.arange(v.shape[0]+1) - 0.5
	x   = numpy.linspace(0, v.shape[0], v.shape[0]+1)
	res = numpy.empty(numpy.array(quantiles, copy=False).shape +
                          (v.shape[1],), dtype=float)
	norm = x*1e-10
	for k in xrange(v.shape[1]):
		# Construct interpolation object
		y  = c[:, k] + norm
                # this tiny fudge ensures y is always monotonically increasing
		res[:, k] = numpy.interp(quantiles, y, x)

	if rescale:
		res = rescale[0] + (rescale[1] - rescale[0]) * (res/v.shape[0])

        if axis != 0:
            v = v.transpose()

        res[:, cs[-1,:] == 0] = numpy.nan

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
    
    (ql, med, qu) = mquantiles(v, (0.25, 0.5, 0.75))
    siqr = 0.5*(qu - ql)

    vmin = med - siqr * (1.349*clip)
    vmax = med + siqr * (1.349*clip)
    mask = (vmin <= v) & (v <= vmax) & numpy.isfinite(v)
    v = v[mask].astype(numpy.float64)
    if len(v) <= 0:
        pdb.set_trace()
    ret = numpy.mean(v), numpy.std(v)
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
	for c, n in izip(reversed(coords), reversed(shape)):
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
        out = fasthist_aux(out, outdims, xc, first, num, binsz, weight=weightc)
        # = rather than += because fasthist_aux overwrites out
        i += 1
    out = out.reshape(outdims)
    return out

def fasthist_aux(out, outdims, x, first, num, binsz, weight=None):
    loc = [numpy.array(numpy.floor((i-f)//sz)+1, dtype='i4')
           for i,f,sz in zip(x, first, binsz)]
    if weight is None:
        weight = numpy.ones(len(x[0]), dtype='u4')
    loc = [numpy.clip(loc0, 0, n+1, out=loc0) for loc0, n in zip(loc, num)]
    flatloc = ravel_index(loc, outdims)
    m = numpy.isfinite(flatloc)
    return add_arr_at_ind(out, weight[m], flatloc[m])

def make_bins(p, range, bin, npix):
    if bin != None and npix != None:
        print "bin size and number of bins set; ignoring bin size"
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
        self.hist_all = fasthist((x,y), (xpts[0], ypts[0]),
                                 (len(xpts)-1, len(ypts)-1),
                                 (xpts[1]-xpts[0], ypts[1]-ypts[0]),
                                 weight=weight)
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
        # 1:-1 to remove the first last last rows in _x_ from the
        # conditional sums. e.g., everything that falls outside of
        # the x limits.  The stuff falling outside of the y limits
        # still counts
        if self.conditional:
            self.q16, self.q50, self.q84 = \
                    percentile_freq(self.hist_all[1:-1,:], [0.16, 0.5, 0.84],
                                   axis=1, rescale=[self.ye[0], self.ye[-1]])

    def show(self, **kw):
        hist = self.hist
        if self.conditional:
            coltot = (self.coltot + (self.coltot == 0))
        else:
            coltot = numpy.ones(len(self.coltot))
        dispim = hist / coltot.reshape((len(coltot), 1))
        extent = [self.xe[0], self.xe[-1], self.ye[0], self.ye[-1]]
        if 'nograyscale' not in kw or not kw['nograyscale']:
            pyplot.imshow(dispim.T, extent=extent, interpolation='nearest',
                          origin='lower', aspect='auto', **kw)
        if self.conditional:
            xpts2 = (numpy.repeat(self.xpts, 2) + 
                     numpy.tile(numpy.array([-1.,1.])*0.5*self.deltx,
                                len(self.xpts)))
            pyplot.plot(xpts2, numpy.repeat(self.q16, 2), "k",
                        xpts2, numpy.repeat(self.q50, 2), "k",
                        xpts2, numpy.repeat(self.q84, 2), "k")
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
            print "field %s has zero size, skipping" % f
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
        print 'Duplicate names.', names[u[nname > 1]]
        raise ValueError('Duplicate column names in table.')
    newrecarray = numpy.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray

# Stolen from MJ
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


def add_arr_at_ind(arr, adds, inds):
    arr = make_normal_arr(arr)
    adds = make_normal_arr(adds)
    inds = make_normal_arr(inds)
    code = """
           for(int i=0; i<Nadds[0]; i++) {
               arr(inds(i)) += adds(i);
           }
           """
    if len(adds) != len(inds):
        raise ValueError('Incompatible argument sizes.')
    if len(adds) == 0:
        return arr
    assert (numpy.min(inds) >= 0 and numpy.max(inds) < len(arr)), \
           'Invalid argument to add_arr_at_ind'
    weave.inline(code, ['arr', 'adds', 'inds'],
                 type_converters=weave.converters.blitz, verbose=1)
    return arr


def pickle_unpickle(x):
    """ Pickles and then unpickles the argument, returning the result.
    Intended to be used to verify that objects pickle successfully."""
    tf = tempfile.TemporaryFile()
    pickle.dump(x, tf)
    tf.seek(0)
    x = pickle.load(tf)
    return x


def make_stats_usable(x):
    """ dumps and loads an ipython stats object to render it usable."""
    tf = tempfile.NamedTemporaryFile()
    x.dump_stats(tf.name)
    return pstats.Stats(tf.name)


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
        pyplot.plot(x, y, markersymbol) ; pyplot.xlim(xrange) ; pyplot.ylim(yrange)
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
        pyplot.plot(x, y, markersymbol) ; pyplot.xlim(xrange) ; pyplot.ylim(yrange)
        return
    if not contour:
        scatterplot(x, y, conditional=False, xrange=xrange, yrange=yrange,
                    norm=norm, **kwargs)
    else:
        contourpts(x, y, xrange=xrange, yrange=yrange, **kwargs)

def djs_iterstat(dat, invvar=None, sigrej=3., maxiter=10.,
                 prefilter=False):
    """ Straight port of djs_iterstat.pro in idlutils"""
    out = { }
    dat = numpy.atleast_1d(dat)
    if invvar is not None:
        invvar = numpy.atleast_1d(invvar)
        assert len(invvar) == len(dat)
        assert numpy.all(invvar >= 0)
    nan = numpy.nan
    ngood = numpy.sum(invvar > 0) if invvar is not None else len(dat)
    if ngood == 0:
        out = {'mean':nan, 'median':nan, 'sigma':nan,
               'mask':numpy.zeros(0, dtype='bool'), 'newivar':nan}
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
    return {'mean':fmean, 'median':fmedian, 'sigma':fsig,
            'mask':savemask, 'newivar':newivar}

def check_sorted(x):
    if len(x) <= 1:
        return True
    return numpy.all(x[1:len(x)] >= x[0:-1])

def solve_lstsq(aa, bb, ivar, return_covar=False):
    d, t = numpy.dot, numpy.transpose
    atcinvb = d(t(aa), ivar*bb)
    atcinva = d(t(aa), ivar.reshape((len(ivar), 1))*aa)
    u,s,vh = numpy.linalg.svd(atcinva)
    par = svsol(u,s,vh,atcinvb)
    var, covar = svd_variance(u, s, vh)
    ret = (par, var)
    if return_covar:
        ret = ret + (covar,)
    return ret

def polyfit(x, y, deg, ivar=None, return_covar=False):
    aa = numpy.zeros((len(x), deg+1))
    for i in xrange(deg+1):
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
    for i in xrange(niter):
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
    return p*180./numpy.pi, 90.-t*180./numpy.pi

def lb2uv(r, d):
    t, p = lb2tp(r, d)
    z = numpy.cos(t)
    x = numpy.cos(p)*numpy.sin(t)
    y = numpy.sin(p)*numpy.sin(t)
    return numpy.vstack([x, y, z]).transpose().copy()

def uv2tp(uv):
    norm = numpy.sqrt(numpy.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = numpy.arccos(uv[:,2])
    p = numpy.arctan2(uv[:,0], uv[:,1])
    return t, p

def xyz2tp(x, y, z):
    norm = numpy.sqrt(x**2+y**2+z**2)
    t = numpy.arccos(z/norm)
    p = numy.arctan2(x/norm, y/norm)
    return t, p

def uv2lb(uv):
    return tp2lb(*uv2tp(uv))

def xyz2lb(x, y, z):
    return tp2lb(*xyz2tp(x, y, z))

def lbr2xyz_galactic(l, b, re, r0=8.5):
    l, b = numpy.radians(l), numpy.radians(b)
    z = re * numpy.sin(b)
    x = r0 - re*numpy.cos(l)*numpy.cos(b)
    y = -re*numpy.sin(l)*numpy.cos(b)
    return x, y, z

def xyz_galactic2lbr(x, y, z, r0=8.5):
    xe = r0-x
    re = numpy.sqrt(xe**2+y**2+z**2)
    b = numpy.degrees(numpy.arcsin(z / re))
    l = numpy.degrees(numpy.arctan2(-y, xe)) % 360.
    return l, b, re

def xyz2rphiz(x, y, z):
    r = numpy.sqrt(x**2+y**2)
    phi = numpy.degrees(numpy.arctan2(y, x))
    return r, phi, z
    
def healgen(nside):
    return healpy.pix2ang(nside, numpy.arange(12*nside**2))

def healgen_lb(nside):
    return tp2lb(*healgen(nside))

def heal_rebin(map, nside, ring=True):
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

def heal2cart(heal, interp=True):
    nside = healpy.get_nside(heal)
    owidth = 8*nside
    oheight = 4*nside-1
    dm,rm = numpy.mgrid[0:oheight,0:owidth]
    rm = 360.-(rm+0.5) / float(owidth) * 360.
    dm = 90. - (dm+0.5) / float(oheight) * 180.
    t, p = lb2tp(rm.flatten(), dm.flatten())
    if interp:
        map = healpy.get_interp_val(heal, t, p)
    else:
        pix = healpy.ang2pix(nside, t, p)
        map = heal[pix]
    map = map.reshape((oheight, owidth))
    return map

def imshow(im, xpts=None, ypts=None, xrange=None, yrange=None, range=None,
           min=None, max=None, mask_nan=False,
           interp_healpy=True, log=False, center_gal=False, color=False,
           **kwargs):
    if xpts is not None and ypts is not None:
        dx = numpy.median(xpts[1:]-xpts[:-1])
        dy = numpy.median(ypts[1:]-ypts[:-1])
        kwargs['extent'] = [xpts[0]-dx/2., xpts[-1]+dx/2.,
                            ypts[0]-dy/2., ypts[-1]+dx/2.]
        if not color:
            im = im.T
        else:
            im = numpy.transpose(im, axes=[0, 1])
        kwargs['origin'] = 'lower'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'binary'
    if isinstance(kwargs['cmap'], str):
        kwargs['cmap'] = matplotlib.cm.get_cmap(kwargs['cmap'])
    oneim = im if not color else im[:,0]
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
        if not center_gal:
            kwargs['extent'] = ((360, 0, -90, 90))
        else:
            if not color:
                im = numpy.roll(im, im.shape[1]/2, axis=1)
            else:
                ncolor = im.shape[-1]
                for i in numpy.arange(ncolor,dtype='i4'):
                    im[:,:,i] = numpy.roll(im[:,:,i], im.shape[1]/2, axis=1)
            kwargs['extent'] = ((180, -180, -90, 90))
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
    out = pyplot.imshow(im, **kwargs)
    if xrange is not None:
        pyplot.xlim(xrange)
    if yrange is not None:
        pyplot.ylim(yrange)
    return out

def contourpts(x, y, xnpix=None, ynpix=None, xrange=None, yrange=None,
                normalize=True, xbin=None, ybin=None, log=False,
                levels=None, nlevels=6, logmin=0, symbol=',', minlevel=1,
               logspace=None, nopoints=False,
               **kw):
     sc = Scatter(x, y, xnpix=xnpix, ynpix=ynpix, xbin=xbin, ybin=ybin,
                  xrange=xrange, yrange=yrange, normalize=normalize,
                  conditional=False)
     if levels is None:
         maximage = numpy.float(numpy.max(sc.hist))
         if log:
             if logspace is None:
                 levels = 10.**(logmin + (numpy.arange(nlevels)+1)*
                                (numpy.log10(maximage)-logmin)/nlevels)
             else:
                 levels = 10.**(numpy.log10(maximage)-logspace*numpy.arange(nlevels))
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
    m = (ind > 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], numpy.flatnonzero(m)

def setup_print(size=None, keys=None, **kw):
    params = {'backend': 'ps',
              'axes.labelsize': 12,
              'text.fontsize': 12,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True }
    for key in kw:
        params[key] = kw[key]
    if keys is not None:
        for key in keys:
            params[key] = keys[key]
    from matplotlib import pyplot
    oldparams = dict(pyplot.rcParams.items())
    pyplot.rcParams.update(params)
    if size is not None:
        pyplot.gcf().set_size_inches(*size, forward=True)
    return oldparams

def arrow(x, y, dx, dy, arrowstyle='->', mutation_scale=30, **kw):
    add_patch = pyplot.gca().add_patch
    FancyArrow = matplotlib.patches.FancyArrowPatch
    add_patch(FancyArrow((x,y),(x+dx,y+dy), arrowstyle=arrowstyle,
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
    t, p = lb2tp(l, b)
    pix = healpy.ang2pix(nside, t, p)
    out = numpy.zeros(12*nside**2)
    wmap = numpy.zeros_like(out)
    out = add_arr_at_ind(out, weight*data, pix)
    wmap = add_arr_at_ind(wmap, weight, pix)
    out = out / (wmap + (wmap == 0))
    return out, wmap

def bindata(x, y, dat, weight=None, xnpix=None, ynpix=None, xrange=None,
            yrange=None, xbin=None, ybin=None):
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
    return (hist_all/(whist_all + (whist_all == 0)), whist_all,
            xpts+ybin/2., ypts+ybin/2.)

def showbindata(x, y, dat, xrange=None, yrange=None, min=None, max=None, **kw):
    im, wim, xpts, ypts = bindata(x, y, dat, xrange=xrange, yrange=yrange,
                                  **kw)
    imshow(im, xpts, ypts, xrange=xrange, yrange=yrange, min=min, max=max)

def cg_to_fits(cg):
    dtype = cg.dtype
    for col in dtype.descr:
        colname = col[0]
        format = col[1]
        if format.find('O') != -1:
            cg.drop_column(colname)
            print 'Warning: dropping column %s because FITS does not support objects.' % colname
            continue
        uloc = format.find('u')
        if uloc == -1:
            uloc = format.find('b')
        if uloc == -1:
            continue
        orig = cg[colname]
        newformat = format[:uloc]+'i'+format[uloc+1:]
        new = numpy.array(orig, newformat)
        cg.drop_column(colname)
        cg.add_column(colname, new)
    return cg
            

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

def stirling_approx(n):
    return n*numpy.log(n) - n + numpy.log(n)/2. + numpy.log(2*numpy.pi)/2.

# PS1 lat/lon: 20.71552, -156.169
def rdllmjd2altaz(r, d, lat, lng, mjd):
    lst = mjd2lst(mjd, lng)
    ha = lst*360./24 - r
    d2r = numpy.pi/180.
    # hadec2altaz.pro in idlutils
    sin, cos = numpy.sin, numpy.cos
    sh, ch = sin(ha*d2r),  cos(ha*d2r)
    sd, cd = sin(d*d2r), cos(d*d2r)
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
