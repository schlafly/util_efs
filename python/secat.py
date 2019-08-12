import os
from astropy.io import fits
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


def headers_to_ndarr(headers, exclude_btable=False):
    names = headers[0].keys()
    names = [n.lower() for n in names
             if (n != 'HISTORY' and n[0:4] != 'SLOT' and
                 len(n) > 0 and n != 'DESCRMSK' and n != 'COMMENT')]
    if exclude_btable:
        names = [n.lower() for n in names if 
                 (n.upper() not in ['XTENSION', 'BITPIX', 
                                    'NAXIS', 'NAXIS1', 'NAXIS2',
                                    'PCOUNT', 'GCOUNT', 'TFIELDS']) and
                 (n[0:5].upper() not in ['TTYPE', 'TFORM'])]
    arrays = [numpy.array([h.get(n, -999) for h in headers]) for n in names]
    names = [n.replace('-', '_') for n in names]
    if numpy.any([len(n) == 0 for n in names]):
        pdb.set_trace()
    return fromarrays(arrays, names=names)


def read_byccd_ldac(filename, **kw):
    hdulist = fits.open(filename)
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
            head = fits.Header(txtfile=StringIO(headerstring2))
        elif len(hdu.name) > 3 and hdu.name[-3:] == 'HDR':
            head = hdu.header
        else:
            if hdu.name != 'LDAC_OBJECTS':
                raise ValueError('Unexpected FITS extension name %s' %
                                 hdu.name)
            dat = fitstondarray(hdu.data, lower=True)
            yield process(head, dat, filename, **kw)


def read_byccd(filename, **kw):
    hdulist = fits.open(filename)
    head = None
    dat = None
    primaryhdu = None
    for hdu in hdulist:
        if hdu.name == 'PRIMARY':
            primaryhdu = hdu.header
        else:
            chip, exttype = hdu.name.split('_')
            if exttype == 'HEADER' or exttype == 'HDR':
                head = hdu.header
                headchip = chip
            elif exttype == 'PSF':
                continue
            else:
                assert exttype == 'CATALOG' or exttype == 'CAT'
                assert headchip == chip
                dat = fitstondarray(hdu.data, lower=True)
                yield process(head, dat, filename, primaryheader=primaryhdu,
                              **kw)


def process(head, dat, fn, primaryheader=None, add_useful_fields=True,
            fix_decam_rd=False):
    if add_useful_fields:
        filename = numpy.zeros(len(dat), dtype='a160')
        mjd = numpy.zeros(len(dat), dtype='f8')
        etime = numpy.zeros(len(dat), dtype='f4')
        filterid = numpy.zeros(len(dat), dtype='a1')
        chip_id = numpy.zeros(len(dat), dtype='i4')
        zp = numpy.zeros(len(dat), dtype='f4')
        basename = os.path.basename(fn)
        filename[:] = basename
        mjd[:] = primaryheader['MJD-OBS']
        if 'alpha_j2000' in dat.dtype.names:
            rafield, decfield = 'alpha_j2000', 'delta_j2000'
        else:
            rafield, decfield = 'ra', 'dec'
        airmass = ps.rdm2airmass(dat[rafield], dat[decfield],
                                 mjd,  lat=-30.1697, lon=-70.8065)
        etime[:] = primaryheader['EXPTIME']
        filterid[:] = primaryheader['FILTER'].split()[0]
        chip_id[:] = head['CCDNUM']
        zph = primaryheader.get('MAGZERO', 'INDEF')
        if zph == 'INDEF':
            primaryheader['MAGZERO'] = 'NaN'
            zph = 'NaN'
        zp[:] = float(zph)
        dat = rec_append_fields(dat,
                                ['mjd_obs', 'exptime',
                                 'filterid', 'airmass', 'chip_id',
                                 'zp'],
                                [mjd, etime,
                                 filterid, airmass, chip_id,
                                 zp])
        if isinstance(primaryheader['ra'], str):
            rahms = primaryheader['ra'].split(':')
            radec = (abs(float(rahms[0]))+float(rahms[1])/60. +
                     float(rahms[2])/60./60.)*360./24.
            radec *= numpy.sign(float(rahms[0]))
            decdms = primaryheader['dec'].split(':')
            decdec = (abs(float(decdms[0]))+float(decdms[1])/60. +
                      float(decdms[2])/60./60.)
            decdec *= numpy.sign(float(decdms[0]))
            primaryheader['ra'] = radec
            primaryheader['dec'] = decdec
        primaryheader['CATFNAME'] = basename
    ndarrhead = headers_to_ndarr([head])
    ndarrphead = (headers_to_ndarr([primaryheader])
                  if primaryheader is not None else None)
    return dat, ndarrhead, ndarrphead


def safe_concatenate(catlist):
    if len(catlist) == 0:
        return numpy.array([])
    dtype = catlist[0].dtype
    out = numpy.zeros(len(catlist), dtype)
    for name in dtype.names:
        arr = numpy.array([catlist[i][name][0] if name in catlist[i].dtype.names
                           else 0 for i in xrange(len(catlist))])
        out[name][:] = arr
    return out


def read(filename, add_useful_fields=True):
    # warning!  these files can be big!
    dats = []
    heads = []
    for dat, head, pri in read_byccd(filename,
                                     add_useful_fields=add_useful_fields):
        dats.append(dat)
        heads.append(head)
    if len(dats) > 0:
        return numpy.concatenate(dats), safe_concatenate(heads), pri
    else:
        return None, None


def read_one(filename, add_useful_fields=True):
    dat, hdrs, pri = read(filename, add_useful_fields=add_useful_fields)
    names = [n for n in hdrs.dtype.names]
    names_new = ['head_'+n for n in names]
    hdtype = []
    for i in xrange(len(names)):
        hdtype.append((names_new[i], hdrs[names[i]].dtype.descr[0][1],
                       len(hdrs)))
    expdat = numpy.zeros(1, dtype=pri.dtype.descr+hdtype)
    for n in pri.dtype.names:
        expdat[n] = pri[n]
    for i in xrange(len(names)):
        expdat[names_new[i]] = hdrs[names[i]]
    return [(dat, expdat)]


def combine_ccd_and_pri_decam(ccd, pri):
    from copy import deepcopy
    outdtype = deepcopy(pri.dtype.descr)
    for rec in ccd[0].dtype.descr:
        if len(rec) != 2:
            raise ValueError('complicated CCD header?')
        outdtype.append(('ccd_'+rec[0], '62'+rec[1]))
    out = numpy.zeros(1, dtype=outdtype)
    for f in pri.dtype.names:
        out[f] = pri[f]
    ccdnum = numpy.array([ccd0['ccdnum'][0] for ccd0 in ccd])-1
    ccda = safe_concatenate(ccd)
    for f in ccd[0].dtype.names:
        out['ccd_'+f][0, ccdnum] = ccda[f]
    return out


def decam_read(filename):
    dats = []
    heads = []
    primaryhdr = None
    for dat, head, pri in read_byccd(filename, add_useful_fields=True):
        dats.append(dat)
        heads.append(head)
        primaryhdr = pri
    if len(dats) > 0:
        yield (numpy.concatenate(dats),
               combine_ccd_and_pri_decam(heads, primaryhdr))
    else:
        yield (None, None)


def wise_read(filename):
    # hack!
    modfname = filename.replace('.cat.', '.mod.').replace('/cat/', '/mod/')
    primaryhdr = fits.getheader(modfname, 1)
    primaryhdr['ra'] = primaryhdr['CRVAL1']
    primaryhdr['dec'] = primaryhdr['CRVAL2']
    primaryhdr['mjd_obs'] = (primaryhdr['mjdmin']+primaryhdr['mjdmax'])/2.
    primaryhdr = headers_to_ndarr([primaryhdr])
    dat = fitstondarray(fits.getdata(filename))
    mjds = numpy.zeros(len(dat), dtype='f8')+primaryhdr['mjd_obs']
    dat = rec_append_fields(dat, 'mjd_obs', mjds)
    yield (dat, primaryhdr)


def ls_photom_v1_read(filename):
    if ls_photom_v1_read.ccds is None:
        print('Loading DECaLS CCDs file...')
        ls_photom_v1_read.ccds = numpy.array(
            fits.getdata(os.environ['LS_CCDS']))
        print('Finished loading DECaLS CCDs file.')
    dat = numpy.array(fits.getdata(filename))
    ccds = ls_photom_v1_read.ccds
    basenames = numpy.array([os.path.basename(f).split('.')[0] 
                             for f in ccds['image_filename']])
    imname = os.path.basename(filename).split('-')[0]
    m = numpy.flatnonzero(imname == basenames)
    if len(m) != 1:
        raise ValueError('Weird filename / CCD match! File: %s' % filename)
    drop_columns = ['image_filename', 'image_hdu', 'expid', 'filter', 
                    'nmatch', 'x', 'y', 'ra', 'dec', 'apmag', 'apflux', 
                    'apskyflux', 'apskyflux_perpix', 'radiff', 'decdiff', 
                    'gaia_g']
    keep_columns = [name for name in dat.dtype.names
                    if name.lower() not in drop_columns]
    if len(keep_columns) != len(dat.dtype.names):
        dat = dat[keep_columns].copy()
    from matplotlib.mlab import rec_append_fields
    mjd = ccds[m]['mjd_obs'][0]
    dat = rec_append_fields(dat.copy(), 'mjd_obs',
                            numpy.zeros(len(dat), dtype='f8')+mjd)
    names = [n.lower() for n in dat.dtype.names]
    names[names.index('ra_ps1')] = 'ra'
    names[names.index('dec_ps1')] = 'dec'
    dat.dtype.names = names
    return [(dat, ccds[m].copy())]
ls_photom_v1_read.ccds = None


def ls_photom_v2_read(filename):
    if ls_photom_v2_read.ccds is None:
        print('Loading CCDs file %s...' % os.environ['LS_CCDS'])
        ls_photom_v2_read.ccds = fits.getdata(os.environ['LS_CCDS'])
        print('Finished loading CCDs file.')
    ccds = ls_photom_v2_read.ccds
    dat, hdr = fits.getdata(filename, 1, header=True)
    dat = numpy.array(dat).copy()

    basenames = numpy.array(['/'.join(f.split('/')[-2:]).strip()
                             for f in ccds['image_filename']])
    imname = hdr['FILENAME'].strip()
    s = numpy.argsort(ccds['ccdname'])
    m = numpy.flatnonzero(imname == basenames[s])
    if len(m) != 1:
        raise ValueError('Weird filename / CCD match! File: %s' % filename)
    ra, dec = ccds['ra'][s[m]][0], ccds['dec'][s[m]][0]
    fwhm = ccds['fwhm'][s[m]][0]
    hdr['RA'] = ra
    hdr['DEC'] = dec
    hdr['FWHM'] = fwhm

    dat = rec_append_fields(dat, 'mjd_obs',
                            numpy.zeros(len(dat), dtype='f8')+hdr['MJD-OBS'])
    names = [n.lower() for n in dat.dtype.names]
    dat.dtype.names = names
    phdr = headers_to_ndarr([hdr], exclude_btable=True)
    return [(dat, phdr)]
ls_photom_v2_read.ccds = None
