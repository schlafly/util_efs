import numpy
from numpy import cos, sin

def xy(the, phi, the_0, phi_0):
    return (sin(the)*sin(phi-phi_0),
            sin(the_0)*cos(the)-cos(the_0)*sin(the)*cos(phi-phi_0))

def thephi(x, y, the_0, phi_0):
    #cosc = sin(the_0)*sin(the) + cos(the_0)*cos(the)*cos(phi-phi_0)
    rho = numpy.sqrt(x**2+y**2)
    c = numpy.arcsin(rho)
    the = numpy.arccos(cos(c)*cos(the_0)+y*sin(the_0))
    phi = phi_0 + numpy.arctan2(x*rho, (rho*sin(the_0)*cos(c) - y*cos(the_0)*rho))
    return the, phi


def rotate(l, b, l0, b0, phi0=0.):
    l = numpy.radians(l)
    b = numpy.radians(b)
    l0 = numpy.radians(l0)
    b0 = numpy.radians(b0)
    ce = numpy.cos(b0)
    se = numpy.sin(b0)
    
    cb, sb = numpy.cos(b), numpy.sin(b)
    cl, sl = numpy.cos(l-phi0), numpy.sin(l-phi0)
    
    ra  = numpy.arctan2(cb*cl, sb*ce-cb*se*sl) + l0
    dec = numpy.arcsin(cb*ce*sl + sb*se)
    
    ra = ra % (2*numpy.pi)
    
    ra = numpy.degrees(ra)
    dec = numpy.degrees(dec)
    
    return ra, dec

def circle(r, d, rad, npts=10):
    phi = numpy.linspace(0, 2*numpy.pi, npts)
    theta = numpy.radians(rad)
    return rotate(numpy.degrees(phi), 90.-theta*180./numpy.pi, r, d)
