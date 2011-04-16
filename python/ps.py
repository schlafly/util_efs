filt = ['g','r','i','z','y']
filtind = dict([(f,i) for i,f in enumerate(filt)])
filt2sdss = dict(((f+'.0000', i+1 if i < 4 else 4) for i,f in enumerate(filt)))

md_fieldcenters = {
    'md00':( 10.675,  41.267),
    'md01':( 35.875,  -4.250),
    'md02':( 53.100, -27.800),
    'md03':(130.592,  44.317),
    'md04':(150.000,   2.200),
    'md05':(161.917,  58.083),
    'md06':(185.000,  47.117),
    'md07':(213.704,  53.083),
    'md08':(242.787,  54.950),
    'md09':(334.188,   0.283),
    'md10':(352.312,  -0.433),
    'md11':(270.000,  66.561) }

flags = {
'DEFAULT' 	 : 0x00000000, 	 #: Initial value: resets all bits
'PSFMODEL' 	 : 0x00000001, 	 #: Source fitted with a psf model (linear or non-linear)
'EXTMODEL' 	 : 0x00000002, 	 #: Source fitted with an extended-source model
'FITTED' 	 : 0x00000004, 	 #: Source fitted with non-linear model (PSF or EXT; good or bad)
'FITFAIL' 	 : 0x00000008, 	 #: Fit (non-linear) failed (non-converge, off-edge, run to zero)
'POORFIT' 	 : 0x00000010, 	 #: Fit succeeds, but low-SN, high-Chisq, or large (for PSF -- drop?)
'PAIR'		 : 0x00000020, 	 #: Source fitted with a double psf
'PSFSTAR' 	 : 0x00000040, 	 #: Source used to define PSF model
'SATSTAR' 	 : 0x00000080, 	 #: Source model peak is above saturation
'BLEND' 	 : 0x00000100, 	 #: Source is a blend with other sourcers
'EXTERNALPOS' 	 : 0x00000200, 	 #: Source based on supplied input position
'BADPSF'	 : 0x00000400, 	 #: Failed to get good estimate of object's PSF
'DEFECT'	 : 0x00000800, 	 #: Source is thought to be a defect
'SATURATED' 	 : 0x00001000, 	 #: Source is thought to be saturated pixels (bleed trail)
'CR_LIMIT' 	 : 0x00002000, 	 #: Source has crNsigma above limit
'EXT_LIMIT' 	 : 0x00004000, 	 #: Source has extNsigma above limit
'MOMENTS_FAILURE': 0x00008000, 	 #: could not measure the moments
'SKY_FAILURE' 	 : 0x00010000, 	 #: could not measure the local sky
'SKYVAR_FAILURE' : 0x00020000, 	 #: could not measure the local sky variance
'MOMENTS_SN' 	 : 0x00040000, 	 #: moments not measured due to low S/N
'BIG_RADIUS' 	 : 0x00080000, 	 #: poor moments for small radius, try large radius
'AP_MAGS' 	 : 0x00100000, 	 #: source has an aperture magnitude
'BLEND_FIT' 	 : 0x00200000, 	 #: source was fitted as a blend
'EXTENDED_FIT' 	 : 0x00400000, 	 #: full extended fit was used
'EXTENDED_STATS' : 0x00800000, 	 #: extended aperture stats calculated
'LINEAR_FIT' 	 : 0x01000000, 	 #: source fitted with the linear fit
'NONLINEAR_FIT'  : 0x02000000, 	 #: source fitted with the non-linear fit
'RADIAL_FLUX' 	 : 0x04000000, 	 #: radial flux measurements calculated
'SIZE_SKIPPED' 	 : 0x08000000, 	 #: size could not be determined
'ON_SPIKE' 	 : 0x10000000, 	 #: peak lands on diffraction spike
'ON_GHOST' 	 : 0x20000000, 	 #: peak lands on ghost or glint
'OFF_CHIP' 	 : 0x40000000 	 #: peak lands off edge of chip
}
for x,y in flags.items():
    flags[y] = x

# definition of flags2 (see http://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/wiki/CMF_PS1_V3)
flags2 = {
'DIFF_WITH_SINGLE' : 0x00000001, 	#: diff source matched to a single positive detection
'DIFF_WITH_DOUBLE' : 0x00000002, 	#: diff source matched to positive detections in both images
'MATCHED'	   : 0x00000004 	#: source was supplied at this location from somewhere else (eg, another image, forced photometry location, etc)
}
for x,y in flags2.items():
    flags2[y] = x
