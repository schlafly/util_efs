#!/usr/bin/env python

import argparse, os, keyword, numpy, pdb
#from astropy.io import fits
import fitsio
from lsd import DB

table_def = \
{
    'filters' : { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False },
    'schema' : {
           'main' : {
                   'columns': [ ('_id', 'u8') ],
                   'primary_key' : '_id',
                   'spatial_keys': ('ra', 'dec'),
                   },
           },
}

def fix_names(dtype, ra, dec):
    names = [n.lower() for n in dtype.names]
    if ra is not '':
        names[names.index(ra)] = 'ra'
    if dec is not '':
        names[names.index(dec)] = 'dec'
    for i, n in enumerate(names):
        if keyword.iskeyword(n):
            names[i] = n+'_'
        if '.' in n:
            names[i] = names[i].replace('.', 'p')
    dtype.names = names


def read_file(file, ra, dec, drop_columns=None, reader=None):
    if reader is None:
        dat = numpy.array(fitsio.read(file, 1)[:])
    else:
        dat = reader(file)
    if dat is not None:
        if drop_columns:
            keep_columns = [name for name in dat.dtype.names
                            if name.lower() not in drop_columns]
            if len(keep_columns) != len(dat.dtype.names):
                dat = dat[keep_columns]
        # FITS capitalization problems...
        dtype = dat.dtype
        fix_names(dtype, ra, dec)
        m = ~numpy.isfinite(dat['ra']) | ~numpy.isfinite(dat['dec'])
        if numpy.any(m):
            dat = dat[~m]
            print('fits_importer.read_file: Some NaN ra/dec')
    return dat
        

def import_file(file, table, ra, dec, drop_columns=None, reader=None):
    try:
        dat = read_file(file, ra, dec, drop_columns=drop_columns,
                        reader=reader)
    except IndexError:
        dat = None
        yield (file, 0)
    except Exception as e:
        print 'Could not read file %s' % file
        raise e
    if dat is not None:
        ids = table.append(dat)
        yield (file, len(ids))

def import_fits(db, table, filedir, ra='', dec='', drop_columns=None,
                reader=None):
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

    try:
        firstfile = read_file(file_list[0], ra, dec, drop_columns=drop_columns,
                              reader=reader)
    except Exception as e:
        print('Could not read first file %s' % file_list[0])
        print(file_list[0:10])
        raise e
    dtype = firstfile.dtype
    from copy import deepcopy
    table_def0 = deepcopy(table_def)
    columns = table_def0['schema']['main']['columns']
    for typedesc in dtype.descr:
        name = typedesc[0]
        type = typedesc[1]
        if (type[0] == '<') or (type[0] == '>'):
            type = type[1:]
        if len(typedesc) > 2:
            if len(typedesc[2]) == 1:
                type = ('%d'+type) % typedesc[2][0]
            else:
                type = ('%s'+type) % str(typedesc[2])
            #tdescr += (typedesc[2],)
            # bit of a hack, but doesn't work with the more complicated format
            # specification
        tdescr = (name, type)
        columns.append(tdescr)
    pool = pool2.Pool()
    db = DB(db)
    with db.transaction():
        if not db.table_exists(table):
            table = db.create_table(table, table_def0)
        else:
            table = db.table(table)
        for fn, num in pool.imap_unordered(file_list, import_file,
                                           (table, ra, dec, drop_columns,
                                            reader)):
            print "Imported file %s containing %d entries." % (fn, num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import FITS files to LSD')
    parser.add_argument('--db', '-d', default=os.environ['LSD_DB'])
    parser.add_argument('--ra', default='', help='column in fits file to rename ra')
    parser.add_argument('--dec', default='', help='column in fits file to rename dec')
    parser.add_argument('--list', default=False, action='store_true')
    parser.add_argument('--drop-columns', default='')
    parser.add_argument('table', type=str, nargs=1)
    parser.add_argument('filedir', type=str, nargs=1)
    args = parser.parse_args()
    files = args.filedir[0]
    if args.list:
        files = [s.strip() for s in open(files, 'r').readlines()]
    print(args.drop_columns)
    if len(args.drop_columns) > 0:
        drop_columns = args.drop_columns.split(' ')
        drop_columns = [d.lower() for d in drop_columns]
    else:
        drop_columns = None
    import_fits(args.db, args.table[0], files,
                ra=args.ra, dec=args.dec, drop_columns=drop_columns)
