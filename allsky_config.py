"""
allsky_config.py

Defines important constants and variables for the all-sky pipeline
"""
# -------- Site Parameters -------
TIMEZONE = "Australia/Hobart"
LATITUDE = -42.4311          # in degrees
LONGITUDE = 147.2878         # in degrees
HEIGHT = 646                 # in metres
OBS_NAME = "Greenhill Observatory"

# -------- Mask and Filtering Parameters -------
MASK_X = 250                 # Zenith mask width in x axis
MASK_Y = 250                 # Zenith mask width in y axis
MOON_RADIUS = 100            # Moon mask circular radius
MEDFILT_SIZE = 90            # Median filter window size

# ---- Optical fit parameters ----
F_fit = 1.769061
R_fit = 338.650091
k3_fit = 1.276095e+01
k5_fit = -1.268911e+00

# ---- Image center / geometry ----
xz_fit = 326.415618
yz_fit = 234.568309
theta_fit = 3.132197

# ---- Extinction / photometric parameters ----
MAG_ZEROPOINT = 4.75        # Magnitude Zeropoint
AIRMASS_TERM = -0.27        # The airmass term
EXP_TIME = 120              # default exposure time (seconds)
MAX_EXTINCTION = 5          # Max extinction value for normalization
PHOT_APERTURE = 1.3         # in pixels

# ---- Catalog parameters ----
ALT_MIN = 20                  # in degrees
MAG_LIMIT = 5                 # catalog magnitude cutoff
MATCH_RADIUS = 2              # in pixels
DROP_CLOSE_RADIUS = 1         # Drop stars that are too close in YBSC (in degrees)