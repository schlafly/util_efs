import random
import numpy
from scipy.stats.mstats import mquantiles
from lsd import DB
from lsd import bounds
import os
import h5py
from matplotlib import pyplot
import pdb
from scipy import weave
import cPickle as pickle
import tempfile
import pstats


def sample(obj, n):
    ind = random.sample(xrange(len(obj)),n)
    return obj[ind]



def unique(obj):
    """Gives an array indexing the unique elements in the sorted list obj.

    unique returns ind, an array of booleans for indexing the unique elements
    of the sorted list obj.  ind[i] is True iff i is the index of the last
    element in the group of equal-comparing items in obj.
    """
    nobj = len(obj)
    if nobj == 0:
        return numpy.zeros(0)
    if nobj == 1:
        return numpy.zeros(1)
    out = numpy.zeros(nobj, dtype=numpy.bool)
    out[0:nobj-1] = (obj[0:nobj-1] != obj[1:nobj])
    out[nobj-1] = True
    return numpy.sort(numpy.flatnonzero(out))



def iqr(dat):
    quant = mquantiles(dat, (0.25, 0.75))
    return quant[1]-quant[0]



class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data):
        self.uind = unique(data)
        self.ind = 0
    def __iter__(self):
        return self
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


def flatten(x):

    def get_flat_arrlist(x):
        print "here"
        name, arr = [], []
        for dstr in x.dtype.descr:
            field = dstr[0]
            print field, x[field].dtype, x[field].dtype.subdtype
            if x[field].dtype.names is None:
                name.append(dstr)
                arr.append(x[field])
            else:
                subname, subarr = get_flat_arrlist(x[field])
                print "subname, subarr:", subname, subarr
                name.append(subname)
                arr.append(subarr)
        print name
        return name, arr
                
    dstr, arr = get_flat_arrlist(x)
    return dstr, arr

full_ps1_db = '/raid14/home/mjuric/lsd/full_ps1/lsd/db_20101203.1'
sas_db = '/raid14/home/mjuric/lsd/db_20101203.1'


def query_lsd(querystr, db=None, boundary=None):

    if db is None:
        db = os.environ['LSD_DB']
    if not isinstance(db, DB):
        dbob = DB(db)
    else:
        dbob = db
    if boundary is not None:
        boundary = bounds.make_canonical(boundary)
    query = dbob.query(querystr)
    return query.fetch(bounds=boundary)


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

def writehdf5(dat, filename, dsname=None):
    if dsname == None:
        dsname = '/'
    f = h5py.File(filename, 'w')
    f.create_dataset(dsname, data=dat)
    f.close()
    

def readhdf5(filename, dsname=None):
    if dsname == None:
        return 1  # not done

    f = h5py.File(filename, 'r')
    ds = f.create_dataset(dsname)
    dat = ds[:]
    f.close()
    return dat


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
def clipped_stats(v, clip=3):
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
    v = numpy.asarray(v)
    (ql, med, qu) = mquantiles(v, (0.25, 0.5, 0.75))
    siqr = 0.5*(qu - ql)
    
    vmin = med - siqr * (1.349*clip)
    vmax = med + siqr * (1.349*clip)
    v = v[(vmin < v) & (v < vmax) & numpy.isfinite(v)].astype(numpy.float64)
    
    return numpy.mean(v), numpy.std(v)


class Scatter:
    def __init__(self, x, y, xnpix=None, ynpix=None, xrange=None, yrange=None,
                 xbin=None, ybin=None):
        if xbin != None and xnpix != None:
            print "bin size and number of bins set; ignoring bin size"
            xbin = None
        if ybin != None and ynpix != None:
            ybin = None
            
        self.xrange = (xrange if xrange is not None else
                       [numpy.min(x), numpy.max(x)])
        self.yrange = (yrange if yrange is not None else
                       [numpy.min(y), numpy.max(y)])

        if xbin != None:
            xnpix = numpy.ceil((self.xrange[1]-self.xrange[0])/xbin)
            xpts = self.xrange[0] + numpy.arange(xnpix+1)*xbin
        else:
            self.xnpix = xnpix if xnpix is not None else 0.3*numpy.sqrt(len(x))
            xpts = numpy.linspace(self.xrange[0], self.xrange[1], self.xnpix+1)
            
        if ybin != None:
            ynpix = numpy.ceil((self.yrange[1]-self.yrange[0])/ybin)
            ypts = self.yrange[0] + numpy.arange(ynpix+1)*ybin
        else:
            self.ynpix = ynpix if ynpix is not None else 0.3*numpy.sqrt(len(x))
            ypts = numpy.linspace(self.yrange[0], self.yrange[1], self.ynpix+1)
        
        inf = numpy.array([numpy.inf])
        xpts2, ypts2 = [numpy.concatenate([-inf, i, inf])
                        for i in (xpts, ypts)]
        self.hist_all, self.xe_a, self.ye_a = \
                    numpy.histogram2d(x, y, bins=[xpts2, ypts2])
        self.hist = self.hist_all[1:-1, 1:-1]
        self.xe = self.xe_a[1:-1]
        self.ye = self.ye_a[1:-1]
        self.deltx, self.delty = self.xe[1]-self.xe[0], self.ye[1]-self.ye[0]
        self.xpts = self.xe[0:-1]+0.5*(self.xe[1]-self.xe[0])
        self.ypts = self.ye[0:-1]+0.5*(self.ye[1]-self.ye[0])
        self.q16, self.q50, self.q84 = \
                  percentile_freq(self.hist_all[1:-1,:], [0.16, 0.5, 0.84],
                                  axis=1, rescale=[self.ye[0], self.ye[-1]])
        self.coltot = numpy.sum(self.hist_all[1:-1,:], axis=1)

    def show(self, **kw):
        hist = self.hist
        coltot = (self.coltot + (self.coltot == 0))
        dispim = hist / coltot.reshape((len(coltot), 1))
        pyplot.imshow(dispim.T, extent=[self.xe[0], self.xe[-1],
                                        self.ye[0], self.ye[-1]],
                      interpolation='nearest', origin='lower', aspect='auto',
                      **kw)
        xpts2 = (numpy.repeat(self.xpts, 2) + 
                 numpy.tile(numpy.array([-1.,1.])*0.5*self.deltx,
                            len(self.xpts)))
        pyplot.plot(xpts2, numpy.repeat(self.q16, 2), "k",
                    xpts2, numpy.repeat(self.q50, 2), "k",
                    xpts2, numpy.repeat(self.q84, 2), "k")
        pyplot.xlim(self.xrange)
        pyplot.ylim(self.yrange)


# Stolen from MJ
def scatterplot(x, y, xnpix=None, ynpix=None, xrange=None, yrange=None, **kw):

    scatter = Scatter(x, y, xnpix=xnpix, ynpix=ynpix,
                      xrange=xrange, yrange=yrange)
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
