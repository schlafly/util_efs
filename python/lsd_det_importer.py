#!/usr/bin/env python

import pyfits, argparse, os, keyword, numpy
from lsd import DB
import secat
from matplotlib.mlab import rec_append_fields
import lsd
import lsd.smf

table_def_det = \
{
    'filters' : { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False },
    'schema' : {
           'main' : {
                   'columns': [ ('det_id', 'u8'),
                                ('exp_id', 'u8'),
                                ('cached', 'bool'),
                              ],
                   'primary_key' : 'det_id',
                   'exposure_key': 'exp_id',
                   'spatial_keys': ('ra', 'dec'),
                   'temporal_key': 'mjd_obs',
                   'cached_flag' : 'cached',
                   },
           },
}

table_def_exp = \
{
    'filters' : { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False },
    'schema' : {
           'main' : {
                   'columns': [ ('exp_id', 'u8'),
                                ('cached', 'bool'),
                              ],
                   'primary_key' : 'exp_id',
                   'exposure_key': 'exp_id',
                   'spatial_keys': ('ra', 'dec'),
                   'temporal_key': 'mjd_obs',
                   'cached_flag' : 'cached',
                   },
           },
}



def fix_names(dtype, ra, dec):
    names = [n.lower() for n in dtype.names]
    if ra is not '':
        if 'ra' in names:
            names[names.index('ra')] = 'fits_ra'
        names[names.index(ra)] = 'ra'
    if dec is not '':
        if 'dec' in names:
            names[names.index('dec')] = 'fits_dec'
        names[names.index(dec)] = 'dec'
    for i, n in enumerate(names):
        if keyword.iskeyword(n):
            names[i] = n+'_'
    dtype.names = names

def import_file(filename, loader, tabledet, tableexp, 
                detra, detdec, expra, expdec):
    import pyfits
    loader = secat.read_one
    try:
        for i, (det_dat, exp_dat) in enumerate(loader(filename)):
            if det_dat != 0:
                #exp_dat = combine_ccd_and_pri(exp_dat, pri_dat)
                # FITS capitalization problems...
                fix_names(det_dat.dtype, detra, detdec)
                fix_names(exp_dat.dtype, expra, expdec)
                # pdb.set_trace()
                emptyexpid = numpy.zeros(len(det_dat), dtype='u8')
                det_dat = rec_append_fields(det_dat, ['exp_id'], [emptyexpid])
                expids = tableexp.append(exp_dat)
                det_dat['exp_id'] = expids[0]
                assert len(numpy.unique(expids)) <= 1, \
                    'Should only import one image at a time?!'
                detids = tabledet.append(det_dat)
                yield ((filename, i), len(detids))
            else:
                yield ((filename, i), 0)
    except Exception as e:
        print e
        yield ((filename, -1), 0)

def update_columns_descr(columns, dtype):
    for typedesc in dtype.descr:
        name = typedesc[0]
        type = typedesc[1]
        if (type[0] == '<') or (type[0] == '>'):
            type = type[1:]
        if len(typedesc) > 2:
            type = ('%d'+type) % typedesc[2][0]
            #tdescr += (typedesc[2],)
            # bit of a hack, but doesn't work with the more complicated format
            # specification and FITS binary tables don't support 
            # multidimensional arrays as columns.
        tdescr = (name, type)
        columns.append(tdescr)

def combine_ccd_and_pri(ccd, pri):
    if pri is None:
        return ccd
    from copy import deepcopy
    outdtype = deepcopy(pri.dtype.descr)
    for rec in ccd.dtype.descr:
        if len(rec) != 2:
            raise ValueError('complicated CCD header?')
        outdtype.append(('ccd_'+rec[0], rec[1]))
    out = numpy.zeros(1, dtype=outdtype)
    for f in pri.dtype.names:
        out[f] = pri[f]
    for f in ccd.dtype.names:
        out['ccd_'+f] = ccd[f]
    return out

def load_first_file(filename, loader):
    dets = [ ]
    heads = [ ]
    primaryheader = None
    for det, head, pri in loader(filename):
        dets.append(det)
        heads.append(head)
        primaryheader = pri
    if len(dets) == 0:
        raise ValueError('No data in first file!')
    for det in dets:
        if det.dtype != dets[0].dtype:
            raise ValueError('inconsistent dtype in different CCDs')
    for head in heads:
        for name in head.dtype.names:
            if name not in heads[0].dtype.names:
                raise ValueError('inconsistent dtype in different headers')
    combexp = combine_ccd_and_pri(heads[0], pri)
    return dets[0].dtype, combexp.dtype

def import_files(db, table, filedir, loader, detra='', detdec='', 
                 expra='', expdec=''):
    from lsd import pool2
    import re
    if not isinstance(filedir, list):
        try:
            file_list = os.listdir(filedir)
            file_list = [os.path.join(filedir, f) for f in file_list]
        except:
            file_list = [ filedir ]
    else:
        file_list = filedir
    detdtype, expdtype = load_first_file(file_list[0], loader)
    try:
        detdtype, expdtype = load_first_file(file_list[0], loader)
    except Exception as e:
        print 'Could not read first file %s' % file_list[0]
        print file_list[0]
        raise e
    fix_names(detdtype, detra, detdec)
    fix_names(expdtype, expra, expdec)
    update_columns_descr(table_def_det['schema']['main']['columns'], detdtype)
    update_columns_descr(table_def_exp['schema']['main']['columns'], expdtype)
    pool = pool2.Pool()
    db = DB(db)
    with db.transaction():
        if not db.table_exists(table+'_det'):
            det_tabname = table+'_det'
            exp_tabname = table+'_exp'
            table_def_exp['commit_hooks'] = [ ('Updating neighbors', 1, 'lsd.smf', 'make_image_cache', [det_tabname]) ]
            tabledet = db.create_table(det_tabname, table_def_det)
            tableexp = db.create_table(exp_tabname, table_def_exp)
            db.define_default_join(det_tabname, exp_tabname,
                                   type = 'indirect',
                                   m1   = (det_tabname, "det_id"),
                                   m2   = (det_tabname, "exp_id"),
                                   _overwrite=True,
                                   )

        else:
            tabledet = db.table(table+'_det')
            tableexp = db.table(table+'_exp')
        for (fn, i), num in pool.imap_unordered(file_list, import_file,
                                                (None, tabledet, tableexp, 
                                                 detra, detdec,
                                                 expra, expdec)):
            print "Imported part %d of file %s containing %d entries." % (i, fn, num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import FITS files to LSD')
    parser.add_argument('--db', '-d', default=os.environ['LSD_DB'])
    parser.add_argument('--detra', default='', help='column in det table to rename ra')
    parser.add_argument('--detdec', default='', help='column in det table to rename dec')
    parser.add_argument('--expra', default='', help='column in exp table to rename ra')
    parser.add_argument('--expdec', default='', help='column in exp table to rename dec')
    parser.add_argument('--loader', default='', help='some mechanism for telling us how to load the files')
    parser.add_argument('--list', default=False, action='store_true')
    parser.add_argument('table', type=str, nargs=1)
    parser.add_argument('filedir', type=str, nargs=1)
    args = parser.parse_args()
    files = args.filedir[0]
    if args.list:
        files = open(files, 'r').readlines()
    import_files(args.db, args.table[0], files,
                 detra=args.detra, detdec=args.detdec, 
                 expra=args.expra, expdec=args.expdec, 
                 )
