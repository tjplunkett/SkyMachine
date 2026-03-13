"""
allsky_phot.py

Author: Thomas Plunkett & Will Gaffney
Organisation: University of Tasmania (UTAS)
Date: 10 March 2025

Description
-----------
This module defines functions to perform photometric analysis of all–sky camera images.
Detected sources in an all–sky frame are matched to stars from the Yale Bright
Star Catalog (YBSC), allowing instrumental photometry to be compared with
catalog magnitudes.

Dependencies
------------
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Astropy
- Astroplan
- Photutils
- SciPy

Notes
-----
- Image coordinates are assumed to be in pixel units.
- Catalog magnitudes are assumed to be V-band.
- Atmospheric extinction is modeled as a linear function of airmass.
"""

# Import necessary functions
import cv2
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.stats import sigma_clip, sigma_clipped_stats, SigmaClip

from astroplan import Observer
from allsky_functions import *

from scipy.spatial.distance import cdist
from scipy import stats, interpolate, ndimage
from scipy.spatial import cKDTree

from photutils.centroids import centroid_com
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus, ApertureStats
from photutils.aperture import RectangularAperture,RectangularAnnulus

import warnings
warnings.filterwarnings("ignore")

def match_catalogs(ybsc_sub, allsky_sub, threshold):
    """
    Match catalog stars with detected sources using a KD-tree.

    This method is significantly faster than brute-force matching and
    scales well for large catalogs.

    Parameters
    ----------
    ybsc_sub : pandas.DataFrame
        Subset of the Yale Bright Star Catalog containing predicted
        image positions.

    allsky_sub : pandas.DataFrame
        Detected source catalog from the all-sky image.

    threshold : float
        Maximum matching distance in pixels.

    Returns
    -------
    match_df : pandas.DataFrame
        Dataframe containing matched catalog and detected sources.
    """

    points1 = ybsc_sub[['x_guess', 'y_guess']].to_numpy()
    points2 = allsky_sub[['x_centroid', 'y_centroid']].to_numpy()

    tree = cKDTree(points2)

    dist, idx = tree.query(points1, distance_upper_bound=threshold)

    valid = dist < threshold

    matched1 = points1[valid]
    matched2 = points2[idx[valid]]

    coord_df = pd.DataFrame({'x_centroid': matched2[:, 0], 'y_centroid': matched2[:, 1],\
                             'x_guess': matched1[:, 0], 'y_guess': matched1[:, 1]})

    match_df = coord_df.merge(ybsc_sub, on=['x_guess', 'y_guess'], how='right')

    return match_df

def run_photometry_circ(im_gray, match_df, ap, JD, exp_time):
    """
    Perform circular aperture photometry for matched stars.

    Parameters
    ----------
    im_gray : numpy.ndarray
        Grayscale image array.

    match_df : pandas.DataFrame
        Dataframe containing matched catalog and image coordinates.

    ap : float
        Radius of circular aperture in pixels.

    JD : float
        Julian Date corresponding to the observation time.
    
    exp_time : float
        Exposure time of the image.

    Returns
    -------
    phot_df : pandas.DataFrame
        Dataframe containing photometric measurements including:

        - Flux
        - Background
        - Airmass
        - Centroid position
    """

    sig = SigmaClip(sigma=3.0, maxiters=10)

    # ------------------------------------------------------------------
    # Sanitize image (handle NaN / Inf)
    # ------------------------------------------------------------------
    bad_mask = ~np.isfinite(im_gray)

    # Replace bad values so photutils won't crash
    im_clean = np.nan_to_num(im_gray, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Validate centroid positions
    # ------------------------------------------------------------------
    valid = match_df[['x_centroid', 'y_centroid']].notna().all(axis=1)

    pos = match_df.loc[valid, ['x_centroid', 'y_centroid']].to_numpy()

    # Remove positions within 10 pixels of the image bounds
    h, w = im_clean.shape
    inside = (
        (pos[:, 0] >= 10) & (pos[:, 0] < w - 10) &
        (pos[:, 1] >= 10) & (pos[:, 1] < h - 10)
    )
    pos = pos[inside]

    # ------------------------------------------------------------------
    # Apertures
    # ------------------------------------------------------------------
    aper = CircularAperture(pos, float(ap))
    annuli = CircularAnnulus(pos, r_in=3.0, r_out=8.0)

    # ------------------------------------------------------------------
    # Photometry (use mask)
    # ------------------------------------------------------------------
    phot_table = aperture_photometry(im_clean, aper, mask=bad_mask)
    aperstats = ApertureStats(im_clean, annuli, mask=bad_mask, sigma_clip=sig)

    # ------------------------------------------------------------------
    # Background handling + table updates
    # ------------------------------------------------------------------
    bkg_median = np.array(aperstats.median)
    bkg_median[~np.isfinite(bkg_median)] = np.nanmedian(bkg_median)

    # Calculate flux and background
    flux = phot_table['aperture_sum'] - bkg_median * aper.area
    bkg = bkg_median * aper.area
    phot_table['Flux'] = flux
    phot_table['Bkg'] = bkg
    phot_table['JD'] = JD
    phot_table['ExpTime'] = exp_time
    phot_table = phot_table.to_pandas()

    phot_table = phot_table.rename(columns={'xcenter': 'x_centroid', 'ycenter': 'y_centroid'})

    # Merge back with original dataframe
    phot_df = match_df.merge(phot_table, on=['x_centroid', 'y_centroid'], how='left')
    phot_df = phot_df.drop(columns=['aperture_sum', 'id'], errors='ignore')
    cosz = np.cos(np.deg2rad(phot_df['Zenith']))
    phot_df['Airmass'] = 1.0 / cosz
    
    # If star is not matched, use the guess positions instead
    phot_df['x_centroid'] = phot_df['x_centroid'].fillna(phot_df.get('x_guess'))
    phot_df['y_centroid'] = phot_df['y_centroid'].fillna(phot_df.get('y_guess'))
    phot_df = phot_df.drop(columns=['x_guess', 'y_guess'], errors='ignore')

    return phot_df

def calib_phot(phot_df, exp):
    """
    Determine photometric zeropoint and atmospheric extinction.

    A linear regression is performed between airmass and the difference
    between catalog magnitude and instrumental magnitude.

    Parameters
    ----------
    phot_df : pandas.DataFrame
        Photometry dataframe.

    exp : float
        Exposure time in seconds.

    Returns
    -------
    tuple
        zeropoint : float
        zeropoint_uncertainty : float
        extinction_coefficient : float
        extinction_uncertainty : float
    """

    # Extract data
    x = phot_df["Airmass"]
    y = phot_df["Mag"] + 2.5 * np.log10(phot_df["Flux"] / exp)

    # Sigma clipping
    clipped = sigma_clip(y, sigma=2.5, maxiters=10, masked=True)
    mask = ~clipped.mask

    x_filt = x[mask]
    y_filt = y[mask]

    # Linear regression
    slope, intercept, r_value, p_value, slope_err = stats.linregress(x_filt, y_filt)

    n = len(x_filt)

    # Residuals
    model = slope * x_filt + intercept
    residuals = y_filt - model

    residual_std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
    sigma = np.std(residuals)

    # Intercept uncertainty
    x_mean = np.mean(x_filt)
    intercept_err = residual_std_error * np.sqrt(
        1 / n + (x_mean**2) / np.sum((x_filt - x_mean) ** 2)
    )

    # Print results
    print(f"Slope: {slope}")
    print(f"Uncertainty in slope: {slope_err}")
    print(f"Intercept: {intercept}")
    print(f"Uncertainty in intercept: {intercept_err}")
    print(f"Sigma: {sigma}")

    zeropoint = intercept
    extinction = slope

    # Plot model line
    x_model = np.arange(x.min(), x.max(), 0.05)
    y_model = zeropoint - abs(extinction) * x_model

    # Plot data
    title = "All Sky Photometry - " + cfg.OBS_NAME
    plt.scatter(x, y, alpha=0.1, label="Outliers")
    plt.scatter(x_filt, y_filt, label="Sigma-clipped data")
    plt.plot(x_model, y_model, linestyle="dashed", label="Model")

    plt.xlabel("Airmass", fontsize=14)
    plt.ylabel("V - $m_{inst}$", fontsize=14)
    plt.title(title, fontweight="bold")

    plt.grid(alpha=0.1)
    plt.legend(loc="upper right")

    plt.show()

    return zeropoint, intercept_err, extinction, slope_err

def make_exmap(file, im_gray, phot_df, zp, k, exp, max_ext, save=False):
    """
    Generate a spatial extinction map across the all-sky image.

    The extinction is computed from the difference between observed
    instrumental magnitude and expected catalog magnitude.

    Parameters
    ----------
    file : str
        Input image filename.

    im_gray : ndarray
        Grayscale all-sky image.

    phot_df : pandas.DataFrame
        Photometry results.

    zp : float
        Photometric zeropoint.

    k : float
        Atmospheric extinction coefficient.

    exp : float
        Exposure time in seconds.

    max_ext : float
        Maximum extinction value used for normalization.

    save : bool, optional
        If True, save extinction map and photometry table.

    Returns
    -------
    float
        Median extinction value for the frame.
    """

    flux = phot_df['Flux'].to_numpy()
    inst_mag = -2.5 * np.log10(flux / exp)

    phot_df['Extinction'] = (
        (zp + inst_mag - abs(k) * phot_df['Airmass'])
        - phot_df['Mag']
    )

    phot_df['Extinction'] = phot_df['Extinction'].fillna(max_ext)

    grid_x, grid_y = np.mgrid[0:im_gray.shape[1], 0:im_gray.shape[0]]

    grid_extinction = interpolate.griddata(points=(phot_df['x_centroid'].to_numpy(),\
                                                   phot_df['y_centroid'].to_numpy()),\
                                           values=phot_df['Extinction'].to_numpy(),\
                                           xi=(grid_x, grid_y), method='nearest')

    nan_mask = np.isnan(grid_extinction)

    if np.any(nan_mask):
        grid_extinction[nan_mask] = interpolate.griddata(points=(phot_df['x_centroid'].to_numpy(),\
                                                                 phot_df['y_centroid'].to_numpy()),\
                                                         values=phot_df['Extinction'].to_numpy(),\
                                                         xi=(grid_x[nan_mask], grid_y[nan_mask]),\
                                                         method='nearest')

    grid_extinction[grid_extinction < 0] = 0

    grid_extinction_smooth = ndimage.median_filter(grid_extinction, size=90)

    grid_z = (10 ** (grid_extinction_smooth / 2.5)) / (10 ** (max_ext / 2.5))
    
    # Save out the extinction maps and photometry, if requested
    if save:

        # Make background more visible
        plt.imshow(im_gray, cmap='gray', vmin=np.percentile(im_gray, 5),
                   vmax=np.percentile(im_gray, 95), alpha=1.0)

        # Make scatter less dominant
        plt.scatter(phot_df['x_centroid'], phot_df['y_centroid'],
                    c=(10 ** (phot_df['Extinction'] / 2.5)) / (10 ** (max_ext / 2.5)),
                    vmin=0.0, vmax=1.0,
                    cmap='viridis',
                    edgecolor='none',
                    s=15,
                    alpha=0.6)

        # Make interpolation overlay faint
        plt.pcolor(grid_x, grid_y, grid_z,
                   vmin=0.0, vmax=1.0,
                   shading='auto',
                   cmap='viridis',
                   alpha=0.2)

        plt.colorbar(label='Extinction Index')

        plt.savefig(file.replace('.jpg', '_ext.jpg'), dpi=300)
        plt.close()

        phot_df.to_csv(file.replace('.jpg', '_phot.csv'))
        
    return np.nanmedian(grid_extinction_smooth)