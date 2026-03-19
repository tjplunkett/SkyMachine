"""
allsky_functions.py

Author: Thomas Plunkett

Purpose
-------
Defines commonly used functions for an all-sky camera analysis pipeline.
These utilities support the extraction and processing of frames from
all-sky videos, optical character recognition (OCR) of timestamps,
astronomical coordinate calculations, and basic star detection.

The functions here are used throughout the pipeline for:

 - Frame extraction from video files
 - Image preprocessing (grayscale conversion and masking)
 - Timestamp extraction from images
 - Star detection using photutils
 - Coordinate transformations for catalogue stars
 - Masking bright objects such as the Moon

Dependencies
------------
OpenCV, EasyOCR, NumPy, Astropy, Photutils

Notes
-----
The pipeline assumes an all-sky camera with the zenith near the centre
of the image and uses circular masking to remove the horizon region.
"""

# Import necessary packages
import allsky_config as cfg
import os
import time
from datetime import datetime, timedelta
import cv2
import easyocr
import numpy as np
from photutils.centroids import centroid_com
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.background import Background2D, MedianBackground
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy import units as u
from astropy.time import Time
from astropy.io import fits

# Initialise EasyOCR reader once so it can be reused efficiently
reader = easyocr.Reader(['en'])  # Reads English text from images

def get_frames(path2avi, output):
    """
    Extract individual frames from an all-sky video.

    Each frame in the video is saved as a JPEG image in the same
    directory as the source video.

    Parameters
    ----------
    path2avi : str
        File path to the input all-sky video (.avi).

    Notes
    -----
    Frame filenames are generated sequentially:

        frame_0.jpg
        frame_1.jpg
        frame_2.jpg
        ...

    This function loads the video using OpenCV and iterates until
    no more frames are returned.
    """

    outpath = output

    # Load video file into memory
    capture = cv2.VideoCapture(path2avi)

    frameNr = 0  # Frame counter

    # Loop through video frames
    while True:

        success, frame = capture.read()

        if success:
            # Save frame as JPEG
            cv2.imwrite(
                os.path.join(outpath, f"frame_{frameNr}.jpg"),
                frame
            )
        else:
            # Exit loop when no frames remain
            break

        frameNr += 1

    # Release video from memory
    capture.release()


def convert2gray(image):
    """
    Convert a colour image to grayscale.

    Parameters
    ----------
    image : ndarray
        Input BGR image (OpenCV format).

    Returns
    -------
    im_gray : ndarray
        Grayscale image.

    Notes
    -----
    Grayscale conversion is typically performed before
    background estimation and star detection.
    """

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return im_gray


def create_maskim(image):
    """
    Apply a circular mask centred on the zenith.

    The mask removes the horizon, buildings and trees from the image.

    Parameters
    ----------
    image : ndarray
        Input image to mask.

    Returns
    -------
    result : ndarray
        Masked image with pixels outside the zenith region set to zero.

    Notes
    -----
    The mask is currently a fixed ellipse with radius ≈ 250 pixels.
    This assumes a known camera geometry and may need tuning
    for different systems.
    """

    mask = np.zeros_like(image)
    rows, cols = mask.shape
    mask_x, mask_y = cfg.MASK_X, cfg.MASK_Y

    # Draw filled ellipse centred on the image
    mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(mask_x, mask_y),\
                       angle=0, startAngle=0, endAngle=360, color=[255, 255, 255],\
                       thickness=-1)

    # Apply mask
    result = np.bitwise_and(image, mask)

    return result, mask


def get_modtime(path):
    """
    Retrieve file modification time as an approximate timestamp.

    The modification time of the video file is used to constrain the
    observation time. This is required due to issues with the text reading.

    Parameters
    ----------
    path : str
        Path to the all-sky video file.

    Returns
    -------
    date_stamp : datetime.date
        Date of the file modification time.

    time_stamp : datetime.time
        Time of the file modification.
    """

    # Modification time in seconds since epoch
    ti_m = os.path.getmtime(path)

    # Convert to readable timestamp
    m_ti = time.ctime(ti_m)

    # Convert to datetime object
    dt = datetime.fromtimestamp(ti_m)

    date_stamp = dt.date()
    time_stamp = dt.time()

    return date_stamp, time_stamp

def get_imtext(image, date_stamp, time_stamp):
    """
    Extract time, date and exposure information from an image.

    EasyOCR is used to read the text overlay typically present
    on all-sky camera frames (date, time, exposure).

    Parameters
    ----------
    image : ndarray
        Image frame containing timestamp text.

    date_stamp : datetime.date
        Date obtained from file modification time.

    time_stamp : datetime.time
        Time obtained from file modification time.

    Returns
    -------
    date_str : str or NaN
        Estimated observation date.

    time_str : str or NaN
        Estimated observation time read from the image.

    exp_str : str or NaN
        Exposure time string (often unreliable).

    Notes
    -----
    Only the upper-left region of the image is analysed because
    the timestamp overlay is expected there.
    """

    # Read text from top-left corner
    text_result = reader.readtext(image[0:65, 0:65],  allowlist="0123456789:/.-s")

    try:
        time_str = text_result[1][1]
    except:
        time_str = None

    try:
        # Determine correct observation date
        # Handles cases where recordings span midnight.

        if 0 <= time_stamp.hour < 12:

            if 0 <= int(time_str[0:2]) <= 12:
                date_str = date_stamp.isoformat()

            elif 16 <= int(time_str[0:2]) <= 23:
                date_str = (date_stamp - timedelta(days=1)).isoformat()

        elif 12 <= time_stamp.hour <= 23:
            date_str = date_stamp.isoformat()

    except:
        date_str = None

    try:
        exp_str = text_result[2][1]
    except:
        exp_str = None

    return date_str, time_str, exp_str


def star_detection(gray_im, masked_im):
    """
    Detect stars in an all-sky image using local background estimation.

    A spatially varying background model is created using
    photutils.Background2D. This allows the detection threshold
    to adapt to sky brightness gradients caused by moonlight,
    airglow, clouds, or horizon light pollution.

    Parameters
    ----------
    gray_im : ndarray
        Unmasked grayscale image used for background estimation.

    masked_im : ndarray
        Masked image used for source detection. This image should
        already have the horizon and moon removed (pixel value = 0).

    Returns
    -------
    tbl : astropy.table.Table
        Table of detected stars containing peak coordinates and
        peak brightness values.

    Notes
    -----
    Image properties for this system:
        - Image size: 450 × 620 pixels
        - Pixel scale: 0.32° / pixel
        - Star size: ~1–2 pixels

    Background modelling parameters:
        box_size = (40,40)

        filter_size = (3,3)
            Smooths the background map to remove noise.

    Detection threshold:
        threshold = 3 × background RMS

    Masking:
        Pixels with value = 0 are excluded from the background
        estimation to avoid bias from horizon or moon masks.
    """

    # Background estimator
    bkg_estimator = MedianBackground()

    # Mask zero-value pixels (these correspond to masked regions)
    mask = masked_im == 0

    # Estimate spatially varying background
    bkg = Background2D(
        gray_im,
        box_size=(40, 40),
        filter_size=(3, 3),
        mask=mask,
        bkg_estimator=bkg_estimator
    )

    # Subtract background from detection image
    data_sub = masked_im - bkg.background

    # Local detection threshold map
    threshold = 3 * bkg.background_rms

    # Detect peaks corresponding to stars, small detection box since stars are ~1–2 pixels
    tbl = find_peaks(data_sub, threshold, box_size=3, centroid_func=centroid_com)

    return tbl.to_pandas(), bkg


def calc_AltAz(ybsc_df, time_str):
    """
    Compute altitude and azimuth for catalogue stars.

    Converts Right Ascension and Declination from the
    Yale Bright Star Catalogue (YBSC) into Alt/Az
    coordinates for the observing site.

    Parameters
    ----------
    ybsc_df : pandas.DataFrame
        Dataframe containing star catalogue data with
        'RA' and 'DEC' columns in degrees.

    time_str : str
        Observation timestamp.

    Returns
    -------
    ybsc_df : pandas.DataFrame
        Updated dataframe including:

        Alt : altitude (degrees)
        Az : azimuth (degrees)
        Zenith : zenith angle (degrees)

    Notes
    -----
    Observatory coordinates are defined in `allsky_config.py`.
    """

    observing_location = EarthLocation(lat=cfg.LATITUDE * u.deg,\
                                       lon=cfg.LONGITUDE * u.deg,\
                                       height=cfg.HEIGHT * u.m)

    observing_time = Time(time_str)

    aa = AltAz(location=observing_location, obstime=observing_time)

    coords = SkyCoord(ybsc_df['RA'], ybsc_df['DEC'], frame='icrs', unit="deg")

    # Transform catalogue coordinates to AltAz
    altaz_coords = coords.transform_to(aa)

    azimuth = altaz_coords.az.deg
    altitude = altaz_coords.alt.deg

    # Zenith angle
    zenith = 90 - np.array(altitude)

    ybsc_df['Alt'] = altitude
    ybsc_df['Az'] = azimuth
    ybsc_df['Zenith'] = zenith

    return ybsc_df

def mask_moon(image):
    """
    Identify and mask the Moon in an all-sky image.

    The Moon is assumed to be the brightest object in the image.
    A blurred version of the image is used to locate the brightest
    pixel, which is then masked with a circular region.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    no_moon : ndarray
        Image with the Moon region removed.

    Notes
    -----
    Steps:
        1. Apply Gaussian blur to suppress noise.
        2. Find brightest pixel (assumed Moon centre).
        3. Create circular mask around this point.
        4. Invert mask and apply to image.
        5. Apply zenith mask.
    """

    # Smooth image to reduce noise
    blur = cv2.GaussianBlur(image, (25, 25), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

    # Moon mask
    moon_mask = np.zeros_like(blur)
    moon_mask = cv2.circle(moon_mask, maxLoc, cfg.MOON_RADIUS, (255, 255, 255), -1)
    moon_mask = cv2.bitwise_not(moon_mask)

    # Zenith mask (from original image)
    _, zenith_mask = create_maskim(image)

    # Combine masks
    full_mask = np.bitwise_and(zenith_mask, moon_mask)

    # Apply combined mask
    no_moon = np.bitwise_and(image, full_mask)

    return no_moon, full_mask

def fix_exp(exp_str):
    """
    Convert exposure time string from image metadata into a float.

    Parameters
    ----------
    exp_str : str
        Exposure time string read from image metadata.

    Returns
    -------
    float
        Exposure time in seconds. Falls back to cfg.EXP_TIME if parsing fails.
    """
    default_exp = float(cfg.EXP_TIME)

    if not exp_str:
        return default_exp
    
    # remove trailing 's' if present
    exp_str = str(exp_str).strip().replace('s', '')  # remove trailing 's' if present

    # Need to check string was extracted correctly and can be converted to a float
    # Sometimes numbers are misinterpreted as letters
    if '.' in exp_str:
        parts = exp_str.split('.', 1)
        t1_str, t2_str = parts[0], parts[1]
        try:
            t1 = float(t1_str)
            try:
                t2 = float("0." + str(t2_str))  # interpret as decimal
                exp = t1 + t2
            except:
                exp = t1  # fallback to integer part only
        except:
            exp = default_exp
    else:
        # no decimal point, try parsing whole string
        try:
            exp = float(exp_str)
        except:
            exp = default_exp

    if exp > default_exp:
        exp = default_exp

    return exp

def make_header(primary_hdu, timestamp,  moon_alt, moon_ill):
    """
    Populate the FITS header of a primary HDU with observatory, image,
    photometric, and optional lunar metadata.

    Parameters
    ----------
    primary_hdu : astropy.io.fits.PrimaryHDU
        The primary HDU object whose header will be updated.
    moon_alt : astropy.units.Quantity or None
        Altitude of the Moon. If None, Moon-related header keywords
        will not be added.
    moon_ill : float or None
        Fractional illumination of the Moon (0 to 1). Used only if
        `moon_alt` is provided.

    Returns
    -------
    primary_hdu : astropy.io.fits.PrimaryHDU
        The updated HDU with new header entries.

    Notes
    -----
    Header categories added:
    - Site properties (location and observatory info)
    - Image properties (exposure, type, masking)
    - Photometric properties (background, zeropoint, extinction)
    - Moon properties (altitude, illumination, mask radius; if available)
    """
    
    # Site properties
    primary_hdu.header['OBSNAME'] = (cfg.OBS_NAME, 'Observatory name')
    primary_hdu.header['LATITUDE'] = (cfg.LATITUDE, 'Observatory latitude in degrees (North positive)')
    primary_hdu.header['LONGTUDE'] = (cfg.LONGITUDE, 'Observatory longitude in degrees (East positive)')
    primary_hdu.header['HEIGHT'] = (cfg.HEIGHT, 'Observatory height in metres')
    
    # Image properties
    primary_hdu.header['DATE-OBS'] = (timestamp, 'Timestamp for image (in iso format)')
    primary_hdu.header['MASKX'] = (cfg.MASK_X, 'Zenith mask width in x-axis')
    primary_hdu.header['MASKY'] = (cfg.MASK_Y, 'Zenith mask width in y-axis') 
    
    # Photometric properties
    primary_hdu.header['MAGZP'] = (cfg.MAG_ZEROPOINT, 'Magnitude zeropoint') 
    primary_hdu.header['XTERM'] = (cfg.AIRMASS_TERM, 'Airmass extinction coefficient') 
    
    # Moon properties
    if moon_alt is not None:
        primary_hdu.header['MOONALT'] = (moon_alt.value, 'Moon Altitude in degrees')
        primary_hdu.header['MOONILL'] = (moon_ill, 'Moon illumination (0 to 1)')
        primary_hdu.header['MOONRAD'] = (cfg.MOON_RADIUS, 'Radius of moon mask')
        
    primary_hdu.header['COMMENT'] = 'Produced by SkyMachine v1 on {:}'.format(str(datetime.now()))
        
    return primary_hdu
        