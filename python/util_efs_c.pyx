import numpy
cimport numpy
cimport cython

ctypedef fused number:
    float
    double
    short
    int
    long
    unsigned short
    unsigned int
    unsigned long

ctypedef fused number2:
    float
    double
    short
    int
    long
    unsigned short
    unsigned int
    unsigned long

ctypedef fused integral:
    short
    int
    long
    unsigned short
    unsigned int
    unsigned long


@cython.boundscheck(False)
@cython.wraparound(False)
def max_bygroup(numpy.ndarray[number] val,
                numpy.ndarray[integral] ind):
    """Return maximum of val array within groups indicated by ind.
    
    ind marks the last element in each group.  The returned array has
    one element per group, equal to the largest value of val in that
    group.

    Args:
        val: the array of values over which the maximum is to be found
        ind: the indices of the last element in each group.  Each group i
             is defined as starting at ind[i-1]+1 and ending at ind[i].

    Returns:
        the maxmimum value in each group.
    """
    if numpy.max(ind) > len(val):
        return ValueError('Incompatible argument sizes.')
    if len(ind) == 0:
        return numpy.array([], dtype=val.dtype)
    if min(ind) < 0:
        raise ValueError('min(ind) must be >= 0')
    

    cdef int pind = 0
    cdef int i, j
    cdef number maxval = val[pind]
    cdef numpy.ndarray[number] out = numpy.zeros(len(ind), dtype=val.dtype)

    for i in range(ind.shape[0]):
        maxval = val[pind]
        for j in range(pind+1, ind[i]+1):
            maxval = maxval if maxval > val[j] else val[j]
        out[i] = maxval
        pind = ind[i]+1
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def add_arr_at_ind(numpy.ndarray[number] arr,
                   numpy.ndarray[number2] adds,
                   numpy.ndarray[integral] inds):
    # in place bincount?
    # add_arr_at_ind(x, y, z) => x += bincount(z, weight=y, minlength=len(x))

    if len(adds) != len(inds):
        raise ValueError('Incompatible argument sizes.')
    if len(adds) == 0:
        return arr

    cdef int i
    cdef int narr = len(arr)
    cdef int nadds = len(adds)
    for i in range(len(inds)):
        if (inds[i] < 0) or (inds[i] >= narr):
            raise ValueError('invalid indices to add_arr_at_ind, %d %d' % 
	                     (inds[i], narr))

    for i in range(nadds):
        arr[inds[i]] += <number> adds[i]

    return arr
