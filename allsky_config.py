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

# -------- Mask Parameters -------
MASK_X = 250
MASK_Y = 250

# ---- Optical fit parameters ----
F_fit = 1.762770
R_fit = 338.324705
k3_fit = 11.833700
k5_fit = -0.986288

# ---- Image center / geometry ----
xz_fit = 326.464569
yz_fit = 234.391257
theta_fit = 3.132448

# ---- Extinction / photometric parameters ----
MAG_ZEROPOINT = 4.75        # Magnitude Zeropoint
AIRMASS_TERM = -0.27        # The airmass term
EXP_TIME = 120              # default exposure time (seconds)
MAX_EXTINCTION = 5.0        # Max extinction value for normalization

# ---- Catalog filtering ----
ALT_MIN = 20                  # in degrees
MAG_LIMIT = 5.0               # catalog magnitude cutoff
MATCH_RADIUS = 2.5            # in pixels
PHOT_APERTURE = 1.3           # in pixels
DROP_CLOSE_RADIUS = 1         # Drop stars that are too close in YBSC (in degrees)