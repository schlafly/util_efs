#!/usr/bin/env python

import pyfits, argparse, os, keyword, numpy
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
    dtype.names = names

def import_file(file, table, ra, dec):
    import pyfits
    try:
        dat = numpy.array(pyfits.getdata(file, 1)[:])
    except IndexError:
        dat = 0
        yield (file, 0)
    except Exception as e:
        print 'Could not read file %s' % file
        raise e
    if dat != 0:
        # FITS capitalization problems...
        dtype = dat.dtype
        fix_names(dtype, ra, dec)
        #pdb.set_trace()
        ids = table.append(dat)
        yield (file, len(ids))

def import_fits(db, table, filedir, ra='', dec=''):
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
        firstfile = pyfits.getdata(file_list[0], 1)
    except Exception as e:
        print 'Could not read first file %s' % file_list[0]
        print file_list[0:10]
        raise e
    dtype = firstfile.dtype
    fix_names(dtype, ra, dec)
    columns = table_def['schema']['main']['columns']
    for typedesc in dtype.descr:
        name = typedesc[0]
        type = typedesc[1]
        if (type[0] == '<') or (type[0] == '>'):
            type = type[1:]
        if len(typedesc) > 2:
            type = ('%d'+type) % typedesc[2][0]
            #tdescr += (typedesc[2],)
            # bit of a hack, but doesn't work with the more complicated format
            # specification and FITS binary tables don't support multidimensional
            # arrays as columns.
        tdescr = (name, type)
        columns.append(tdescr)
    pool = pool2.Pool()
    db = DB(db)
    with db.transaction():
        if not db.table_exists(table):
            table = db.create_table(table, table_def)
        else:
            table = db.table(table)
        for fn, num in pool.imap_unordered(file_list, import_file,
                                           (table, ra, dec)):
            print "Imported file %s containing %d entries." % (fn, num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import FITS files to LSD')
    parser.add_argument('--db', '-d', default=os.environ['LSD_DB'])
    parser.add_argument('--ra', default='', help='column in fits file to rename ra')
    parser.add_argument('--dec', default='', help='column in fits file to rename dec')
    parser.add_argument('--list', default=False, action='store_true')
    parser.add_argument('table', type=str, nargs=1)
    parser.add_argument('filedir', type=str, nargs=1)
    args = parser.parse_args()
    files = args.filedir[0]
    if args.list:
        files = open(files, 'r').readlines()
    import_fits(args.db, args.table[0], files,
                ra=args.ra, dec=args.dec)
