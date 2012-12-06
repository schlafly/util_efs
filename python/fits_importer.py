#!/usr/bin/env python

import pyfits, argparse, os
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


def import_file(file, table):
    import pyfits
    try:
        dat = pyfits.getdata(file, 1)
    except Exception as e:
        print 'Could not read file %s' % file
        raise e
    # FITS capitalization problems...
    dtype = dat.dtype
    dat.dtype.names = [n.lower() for n in dat.dtype.names]
    #pdb.set_trace()
    ids = table.append(dat)
    yield (file, len(ids))

def import_fits(db, table, filedir):
    from lsd import pool2
    import numpy, re
    if not isinstance(filedir, list):
        file_list = os.listdir(filedir)
        file_list = [os.path.join(filedir, f) for f in file_list]
    else:
        file_list = filedir

    firstfile = pyfits.getdata(file_list[0], 1)
    dtype = firstfile.dtype
    columns = table_def['schema']['main']['columns']
    for typedesc in dtype.descr:
        name = typedesc[0].lower()
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
        for fn, num in pool.imap_unordered(file_list, import_file, (table,)):
            print "Imported file %s containing %d entries." % (fn, num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import FITS files to LSD')
    parser.add_argument('--db', '-d', default=os.environ['LSD_DB'])
    parser.add_argument('table', type=str, nargs=1)
    parser.add_argument('filedir', type=str, nargs=1)
    args = parser.parse_args()
    import_fits(args.db, args.table[0], args.filedir[0])
