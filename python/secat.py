import os
import pyfits
from matplotlib.mlab import rec_append_fields, rec_drop_fields
from numpy.core.records import fromarrays
import numpy
from StringIO import StringIO
import ps
import pdb

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

def headers_to_ndarr(headers):
    names = headers[0].keys()
    names = [n.lower() for n in names 
             if (n != 'HISTORY' and n[0:4] != 'SLOT' and 
                 len(n) > 0 and n != 'DESCRMSK')]
    arrays = [ numpy.array([h.get(n, -999) for h in headers]) for n in names ]
    names = [n.replace('-', '_') for n in names]
    if numpy.any([len(n) == 0 for n in names]):
        pdb.set_trace()
    return fromarrays(arrays, names=names)

def read_byccd_ldac(filename, add_useful_fields=True):
    hdulist = pyfits.open(filename)
    head = None
    dat = None
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            continue
        elif hdu.name == 'LDAC_IMHEAD':
            headerstring = hdu.data['Field Header Card'][0]
            # FITS header has 80 character lines, closing 3 character END
            lines = [headerstring[i:i+80] 
                     for i in xrange(0, len(headerstring)-3, 80)]
            lines.append('')
            headerstring2 = '\n'.join(lines)
            head = pyfits.Header(txtfile=StringIO(headerstring2))
        else:
            if hdu.name != 'LDAC_OBJECTS':
                raise ValueError('Unexpected FITS extension name %s' % hdu.name)
            dat = fitstondarray(hdu.data, lower=True)
            yield process(head, dat, filename, 
                          add_useful_fields=add_useful_fields)

def read_byccd(filename, add_useful_fields=True):
    hdulist = pyfits.open(filename)
    head = None
    dat = None
    primaryhead = None
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            primaryhdu = hdu.header
        else:
            chip, exttype = hdu.name.split('_')
            if exttype == 'HEADER':
                head = hdu.header
                headchip = chip
            else:
                assert exttype == 'CATALOG'
                assert headchip == chip
                dat = fitstondarray(hdu.data, lower=True)
                yield process(head, dat, filename, primaryheader=primaryhdu,
                              add_useful_fields=add_useful_fields)


def process(head, dat, fn, primaryheader=None, add_useful_fields=True):
    if add_useful_fields:
        filename = numpy.zeros(len(dat), dtype='a160')
        mjd = numpy.zeros(len(dat), dtype='f8')
        etime = numpy.zeros(len(dat), dtype='f4')
        filterid = numpy.zeros(len(dat), dtype='a80')
        chip_id = numpy.zeros(len(dat), dtype='i4')
        zp = numpy.zeros(len(dat), dtype='f4')
        basename = os.path.basename(fn)
        filename[:] = basename
        mjd[:] = primaryheader['MJD-OBS']
        airmass = ps.rdm2airmass(dat['alpha_j2000'], dat['delta_j2000'], 
                                 mjd,  lat=-30.1697, lon=-70.8065)
        etime[:] = primaryheader['EXPTIME']
        filterid[:] = primaryheader['FILTER']
        chip_id[:] = head['CCDNUM']
        zph = primaryheader['MAGZERO']
        if zph == 'INDEF':
            primaryheader['MAGZERO'] = 'NaN'
            zph = 'NaN'
        zp[:] = float(zph)
        dat = rec_append_fields(dat, 
                                ['filename', 'mjd_obs', 'exptime', 
                                 'filterid', 'airmass', 'chip_id',
                                 'zp'],
                                [filename, mjd, etime, 
                                 filterid, airmass, chip_id,
                                 zp])
        if isinstance(primaryheader['ra'], str):
            rahms = primaryheader['ra'].split(':')
            radec = (float(rahms[0])+float(rahms[1])/60.+float(rahms[2])/60./60.)*360./24.
            decdms = primaryheader['dec'].split(':')
            decdec = (float(decdms[0])+float(decdms[1])/60.+float(decdms[2])/60./60.)
            primaryheader['ra'] = radec
            primaryheader['dec'] = decdec
    ndarrhead = headers_to_ndarr([head])
    ndarrphead = headers_to_ndarr([primaryheader]) if primaryheader is not None else None
    return dat, ndarrhead, ndarrphead

def safe_concatenate(catlist):
    if len(catlist) == 0:
        return numpy.array([])
    dtype = catlist[0].dtype
    out = numpy.zeros(len(catlist), dtype)
    for name in dtype.names:
        arr = numpy.array([catlist[i][name][0] for i in xrange(len(catlist))])
        out[name][:] = arr
    return out

def read(filename, add_useful_fields=True):
    # warning!  these files can be big!
    dats  = [ ]
    heads = [ ]
    for dat, head, pri in read_byccd(filename, add_useful_fields=add_useful_fields):
        dats.append(dat)
        heads.append(head)
    return numpy.concatenate(dats), safe_concatenate(heads), pri

def read_one(filename, add_useful_fields=True):
    dat, hdrs, pri = read(filename, add_useful_fields=add_useful_fields)
    names = [n for n in hdrs.dtype.names]
    names_new = ['head_'+n for n in names]
    fields = [hdrs[n] for n in names]
    hdtype = []
    for i in xrange(len(names)):
        hdtype.append( (names_new[i], hdrs[names[i]].dtype.descr[0][1],
                        len(hdrs)) )
    expdat = numpy.zeros(1, dtype=pri.dtype.descr+hdtype)
    for n in pri.dtype.names:
        expdat[n] = pri[n]
    for i in xrange(len(names)):
        expdat[names_new[i]] = hdrs[names[i]]
    return [ (dat, expdat) ]
