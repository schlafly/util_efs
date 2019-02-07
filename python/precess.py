import numpy as np
import util_efs, pdb

def precess(r, d, fromequinox, toequinox):
    eqx = np.zeros(len(r), dtype=[('from', 'f8'), ('to', 'f8')])
    eqx['from'] = fromequinox
    eqx['to'] = toequinox
    s = np.lexsort((eqx['from'], eqx['to']))
    u = util_efs.unique_multikey(eqx[s], keys=['from', 'to'])
    r2 = np.zeros_like(r)
    d2 = np.zeros_like(d)
    for f, l in util_efs.subslices(eqx[s], uind=u):
        ind = s[f:l]
        pmat = precession_matrix_Capitaine(eqx['from'][ind[0]],
                                           eqx['to'][ind[0]])
        uv = util_efs.lb2uv(r[ind], d[ind])
        uv2 = np.array(np.dot(pmat, uv.transpose()).transpose())
        r2[ind], d2[ind] = util_efs.uv2lb(uv2)
    return r2, d2

def rotation_matrix(angle, axis='z', degrees=True):
    """
Generate a 3x3 cartesian rotation matrix in for rotation about
a particular axis.

Parameters
----------
angle : scalar
The amount of rotation this matrix should represent. In degrees
if `degrees` is True, otherwise radians.
axis : str or 3-sequence
Either 'x','y', 'z', or a (x,y,z) specifying an axis to rotate
about. If 'x','y', or 'z', the rotation sense is
counterclockwise looking down the + axis (e.g. positive
rotations obey left-hand-rule).
degrees : bool
If True the input angle is degrees, otherwise radians.

Returns
-------
rmat: `numpy.matrix`
A unitary rotation matrix.
"""
    from numpy import sin, cos, radians, sqrt

    if degrees:
        angle = radians(angle)

    if axis == 'z':
        s = sin(angle)
        c = cos(angle)
        return np.matrix(((c, s, 0),
                          (-s, c, 0),
                          (0, 0, 1)))
    elif axis == 'y':
        s = sin(angle)
        c = cos(angle)
        return np.matrix(((c, 0, -s),
                          (0, 1, 0),
                          (s, 0, c)))
    elif axis == 'x':
        s = sin(angle)
        c = cos(angle)
        return np.matrix(((1, 0, 0),
                          (0, c, s),
                          (0, -s, c)))
    else:
        x, y, z = axis
        w = cos(angle / 2)

        # normalize
        if w == 1:
            x = y = z = 0
        else:
            l = sqrt((x * x + y * y + z * z) / (1 - w * w))
            x /= l
            y /= l
            z /= l

        wsq = w * w
        xsq = x * x
        ysq = y * y
        zsq = z * z
        return np.matrix(((wsq + xsq - ysq - zsq, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y),
                          (2 * x * y + 2 * w * z, wsq - xsq + ysq - zsq, 2 * y * z - 2 * w * x),
                          (2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, wsq - xsq - ysq + zsq)))

def precession_matrix_Capitaine(fromyear, toyear):
    """
Stolen wholesale from astropy
Computes the precession matrix from one julian epoch to another.
The exact method is based on Capitaine et al. 2003, which should
match the IAU 2006 standard.

Parameters
----------
fromepoch : `~astropy.time.Time`
The epoch to precess from.
toepoch : `~astropy.time.Time`
The epoch to precess to.

Returns
-------
pmatrix : 3x3 array
Precession matrix to get from `fromepoch` to `toepoch`

References
----------
USNO Circular 179
"""
    mat_fromto2000 = _precess_from_J2000_Capitaine(fromyear).T
    mat_2000toto = _precess_from_J2000_Capitaine(toyear)

    return np.dot(mat_2000toto, mat_fromto2000)


def _precess_from_J2000_Capitaine(epoch):
    """
Computes the precession matrix from J2000 to the given Julian Epoch.
Expression from from Capitaine et al. 2003 as expressed in the USNO
Circular 179. This should match the IAU 2006 standard from SOFA.

Parameters
----------
epoch : scalar
The epoch as a julian year number (e.g. J2000 is 2000.0)

"""

    T = (epoch - 2000.0) / 100.0
    # from USNO circular
    pzeta = (-0.0000003173, -0.000005971, 0.01801828, 0.2988499, 2306.083227, 2.650545)
    pz = (-0.0000002904, -0.000028596, 0.01826837, 1.0927348, 2306.077181, -2.650545)
    ptheta = (-0.0000001274, -0.000007089, -0.04182264, -0.4294934, 2004.191903, 0)
    zeta = np.polyval(pzeta, T) / 3600.0
    z = np.polyval(pz, T) / 3600.0
    theta = np.polyval(ptheta, T) / 3600.0

    return rotation_matrix(-z, 'z') *\
           rotation_matrix(theta, 'y') *\
           rotation_matrix(-zeta, 'z')

