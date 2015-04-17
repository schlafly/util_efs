# hacked out of lsd.builtins.misc, credit to Mario Juric
def fits_quickparse(header):
    """
    An ultra-simple FITS header parser.

    Does not support CONTINUE statements, HIERARCH, or anything of the
    sort; just plain vanilla:

    key = value / comment

    one-liners. The upshot is that it's fast, much faster than the
    PyFITS equivalent.

    NOTE: Assumes each 80-column line has a '\n' at the end (which is
    how we store FITS headers internally.)
    """
    res = {}
    for line in header.split('\n'):
        at = line.find('=')
        if at == -1:
            continue

        # get key
        key = line[0:at].strip()
        if key[0:7] == 'COMMENT':
            continue
        if key[0:8] == 'HIERARCH':
            key = key[9:]

        # parse value (string vs number, remove comment)
        val = line[at+1:].strip()
        if val[0] == "'":
            # string
            val = val[1:val.find("'", 1)]
        else:
            # number or T/F
            at = val.find('/')
            if at == -1: at = len(val)
            val = val[0:at].strip()
            if val.lower() in ['t', 'f']:
                # T/F
                val = val.lower() == 't'
            else:
                # Number
                val = float(val)
                if int(val) == val:
                    val = int(val)
        res[key] = val
    return res
