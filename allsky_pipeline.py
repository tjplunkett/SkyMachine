"""
allsky_pipeline.py

Author: Thomas Plunkett & Will Gaffney
Organisation: University of Tasmania (UTAS)
Date: 10 March 2025

Description
-----------
All-sky cloud detection pipeline.

This script processes either:
    1) A video file (frames extracted automatically)
    2) A single image (TO DO - still to be implemented)

For each frame it:
    - detects stars
    - matches stars to a catalog
    - performs photometry
    - estimates extinction
    - measures sky background
    - calculates star detection fraction

Parallel processing is used for speed.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import allsky_config as cfg
from allsky_functions import *
from allsky_phot import *
import glob
import os
import time
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib
from datetime import timedelta
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import Observer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import sys
import gc

# ------------------------------------------------------------------
# Optical distortion model
# ------------------------------------------------------------------

def calc_r(z, F, R, k3, k5):
    """
    Compute distorted radial distance in the camera projection.

    Parameters
    ----------
    z : ndarray
        Zenith angle in radians.
    F : float
        Focal length parameter.
    R : float
        Primary radial scaling.
    k3, k5 : float
        Higher-order distortion coefficients.

    Returns
    -------
    ndarray
        Radial distance from image center in pixels.
    """
    return R * np.sin(z / F) + k3 * z**3 + k5 * z**5

# ------------------------------------------------------------------
# Catalog cleaning
# ------------------------------------------------------------------

def drop_close_sources(df, radius_deg=1.0):
    """
    Remove stars that lie too close together, prioritising bright stars.

    Close stars cause blended photometry and unreliable catalog
    matching, so they are removed once during initialization.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog containing columns 'RA' and 'DEC'.
    radius_deg : float
        Minimum allowed separation between stars in degrees.

    Returns
    -------
    pandas.DataFrame
        Catalog with close neighbours removed.
    """

    df = df.sort_values('Mag').reset_index(drop=True)

    coords = SkyCoord(df["RA"].values*u.deg, df["DEC"].values*u.deg)
    radius = radius_deg * u.deg

    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not keep[i]:
            continue

        sep = coords[i].separation(coords)
        close = (sep < radius) & (sep > 0*u.deg)

        keep[close] = False

    return df[keep].reset_index(drop=True)

# ------------------------------------------------------------------
# Global variables shared by workers
# ------------------------------------------------------------------

ybsc_master = None
date_stamp = None
time_stamp = None
observatory_location = EarthLocation(lat=cfg.LATITUDE*u.deg, lon=cfg.LONGITUDE*u.deg,\
                                     height=cfg.HEIGHT*u.m)
OBS = Observer(location=observatory_location, name=cfg.OBS_NAME, timezone=cfg.TIMEZONE)

# ------------------------------------------------------------------
# Worker initialization
# ------------------------------------------------------------------

def init_worker(reference_path):
    """
    Initialize each multiprocessing worker.

    Tasks performed:
        - limit OpenCV threading
        - load star catalog once
        - remove close stars
        - extract timestamp calibration from reference video

    Parameters
    ----------
    reference_path : str
        Path to the video used for timestamp calibration.
    """

    global ybsc_master
    global date_stamp, time_stamp

    # Prevent OpenCV from spawning additional threads
    cv2.setNumThreads(1)
    
    matplotlib.use("Agg")  # safe for multiprocessing

    # Extract timestamp calibration
    date_stamp, time_stamp = get_modtime(reference_path)

    # Load catalog
    ybsc_master = pd.read_csv("Yale_Cat.csv")

    # Remove close sources once
    ybsc_master = drop_close_sources(ybsc_master,cfg.DROP_CLOSE_RADIUS)

# ------------------------------------------------------------------
# Main per-frame processing function
# ------------------------------------------------------------------

def process_file(file, save = False, verbose = False):
    """
    Process a single all-sky image.

    Parameters
    ----------
    file : str
        Path to image file.

    Returns
    -------
    tuple or None
        (extinction_index, julian_date, sky_background, detection_fraction, exp_time)

        Returns None if processing fails.
    """

    global ybsc_master
    global date_stamp, time_stamp

    try:

        # ----------------------------------------------------------
        # Image loading
        # ----------------------------------------------------------

        im = cv2.imread(file)

        if im is None:
            raise ValueError("Image could not be read")

        im_gray = convert2gray(im)

        # ----------------------------------------------------------
        # Image text reading + time extraction
        # ----------------------------------------------------------

        date_str, time_str, exp = get_imtext(im_gray, date_stamp, time_stamp)
        
        if date_str is not None and time_str is not None:
            exp_time = fix_exp(exp)
            time_str = time_str.replace(".", ":")
            time_str = time_str.replace("::", ":")

            # Convert local time to UTC
            t_local = pd.to_datetime(str(date_str) + " " + str(time_str))
            t_str = t_local.tz_localize(cfg.TIMEZONE).tz_convert("UTC")

            astro_time = Time(t_str)
            JD = astro_time.jd

            # ---------------------------------------------------------
            # Moon masking
            # ---------------------------------------------------------

            moon_alt = None
            moon_az = None

            try:
                moon = OBS.moon_altaz(astro_time)

                moon_alt = moon.alt
                moon_az = moon.az
                moon_ill = OBS.moon_illumination(astro_time)

                if moon_alt > 0 * u.deg:
                    mask_im = mask_moon(im_gray)
                else:
                    mask_im = create_maskim(im_gray)

            except Exception as e:
                if verbose:
                    print(f"Moon calculation failed: {e}")
                mask_im = create_maskim(im_gray)

            # ---------------------------------------------------------
            # Source detection
            # ---------------------------------------------------------

            tas_df, bkg = star_detection(im_gray, mask_im)

            bkg_median = np.nanmedian(bkg.background)
            bkg_rms = bkg.background_rms

            if save:
                cv2.imwrite(file.replace('.jpg', '_bkg.jpg'), bkg.background)
                cv2.imwrite(file.replace('.jpg', '_mask.jpg'), mask_im)

            # ----------------------------------------------------------
            # Catalog transformation
            # ----------------------------------------------------------

            ybsc_df = calc_AltAz(ybsc_master.copy(), t_str)

            mask = ((ybsc_df.Alt.values >= cfg.ALT_MIN) & (ybsc_df.Mag.values < cfg.MAG_LIMIT))

            ybsc_sub = ybsc_df.loc[mask]

            # ----------------------------------------------------------
            # Remove stars within 35° of Moon
            # ----------------------------------------------------------

            if moon_alt is not None and moon_alt > 0 * u.deg:

                moon_coord = SkyCoord(alt=moon_alt, az=moon_az, frame="altaz")

                star_coords = SkyCoord(alt=ybsc_sub.Alt.values * u.deg,\
                                       az=ybsc_sub.Az.values * u.deg,\
                                       frame="altaz")

                sep = star_coords.separation(moon_coord)

                moon_mask = sep > 35 * u.deg

                ybsc_sub = ybsc_sub.loc[moon_mask]

            # ----------------------------------------------------------
            # Optical distortion model
            # ----------------------------------------------------------

            r = calc_r(np.deg2rad(ybsc_sub["Zenith"]), cfg.F_fit, cfg.R_fit,\
                       cfg.k3_fit, cfg.k5_fit)

            ybsc_sub["x_guess"] = (
                cfg.xz_fit +
                r * np.cos(np.deg2rad(ybsc_sub["Az"]) - cfg.theta_fit)
            )

            ybsc_sub["y_guess"] = (
                cfg.yz_fit -
                r * np.sin(np.deg2rad(ybsc_sub["Az"]) - cfg.theta_fit)
            )

            # ----------------------------------------------------------
            # Catalog matching
            # ----------------------------------------------------------
            
            # Safety for matching
            tas_df = tas_df.replace([np.inf, -np.inf], np.nan)
            tas_df = tas_df.dropna(subset=['x_centroid', 'y_centroid'])
            match_df = match_catalogs(ybsc_sub, tas_df, cfg.MATCH_RADIUS)

            # ----------------------------------------------------------
            # Photometry - Circular Aperture (best option for UTGO)
            # ----------------------------------------------------------

            phot_df = run_photometry_circ(im_gray, match_df, cfg.PHOT_APERTURE, JD, exp_time)

            # ----------------------------------------------------------
            # Extinction calculation and auxillary measurements
            # ----------------------------------------------------------

            extinction = make_exmap(file, im_gray, phot_df, cfg.MAG_ZEROPOINT,\
                                    cfg.AIRMASS_TERM, float(exp_time), cfg.MAX_EXTINCTION,\
                                    save)
            
            # Fraction of detected stars vs expected stars
            detection_fraction = (len(phot_df.dropna(subset=['Flux'])) / len(ybsc_sub))

            return (extinction, JD, bkg_median, detection_fraction, exp_time)
        
        else:
            if verbose:
                print(f"Failed on {file}: Unable to extract time or date...")
    except:
        if verbose:
            print(f"Failed on {file}")
        return None


# ------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Set-up the parser
    parser = argparse.ArgumentParser(description="Extract cloud cover from all-sky images")

    parser.add_argument("path", type=str, help="Path to video or image")

    parser.add_argument("type", type=str, help="Input type: 'video' or 'image'")

    parser.add_argument("output", type=str, help="Output directory")
    
    parser.add_argument("save", type=str, help="Save extinction maps and photometry? (y/n)")
    
    parser.add_argument("verbose", type=str, help="Do you want printed output? (y/n)")
    
    # --------------------------------------------------------------
    # Extract user options + setup
    # --------------------------------------------------------------
    
    # Only need this for mac - comment out if using other IOS
    mp.set_start_method("spawn", force=True)
    
    args = parser.parse_args()
    nproc = int((mp.cpu_count() / 2) + 1)
                 
    if args.save == 'y' or args.save == 'Y':
        save = True
    else:
        save = False
        
    if args.verbose == 'y' or args.verbose == 'Y':
        verbose = True
    else:
        verbose = False

    # --------------------------------------------------------------
    # Video splitting + File discovery
    # --------------------------------------------------------------

    if args.type == "image":

        files = [args.path]  
        reference_path = args.path

    elif args.type == "video":

        get_frames(args.path)
        files = glob.glob(os.path.join(os.path.dirname(args.path), "*.jpg"))
        reference_path = args.path

    else:
        raise ValueError("type must be 'image' or 'video'")

    # --------------------------------------------------------------
    # Parallel processing + safe cleanup
    # --------------------------------------------------------------

    start_time = time.perf_counter()
    results = []

    executor = ProcessPoolExecutor(
        max_workers=nproc,
        initializer=init_worker,
        initargs=(reference_path,)
    )

    try:
        # Submit all tasks
        futures = {executor.submit(process_file, f, save, verbose): f for f in files}

        for i, future in enumerate(as_completed(futures), 1):

            file = futures[future]

            try:
                r = future.result(timeout=10)

                if r is not None:
                    results.append(r)

                # Optional: print progress
                if verbose:
                    print(f"[{i}/{len(files)}] Finished {file}", flush=True)

                # Explicit memory cleanup
                del r
                gc.collect()

            except TimeoutError:
                if verbose:
                    print(f"TIMEOUT: {file}")

            except:
                if verbose:
                    print(f"FAILED: {file}")

    finally:
        # Ensure all workers are properly terminated
        executor.shutdown(wait=True, cancel_futures=True)
        # Final garbage collection
        gc.collect()

    end_time = time.perf_counter()
    if verbose:
        print(f"Finished in {end_time - start_time:.2f} seconds")
        print(f"Successful: {len(results)}")

    # --------------------------------------------------------------
    # Extract and save results
    # --------------------------------------------------------------
    
    # Safety
    if not results:  
        raise RuntimeError("No frames were processed successfully.")

    ext_vec, t_vec, bkg_vec, frac_vec, exp_vec = zip(*results)

    final_df = pd.DataFrame({
        "JD": t_vec,
        "DetectionFraction": frac_vec,
        "SkyBkg": bkg_vec,
        "ExtinctionIndex": ext_vec,
        "ExpTime": exp_vec
    })

    out_name = os.path.join(args.output, f"CloudDetection_{os.path.basename(args.path)}.csv")

    final_df.to_csv(out_name, index=False)
    
    # Clean up files if not wanting to save
    if args.type == 'video' and save == False:
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                if verbose:
                    print(f"Error removing {file_path}: {e}")
    
    if verbose:
        print("Saved:", out_name)