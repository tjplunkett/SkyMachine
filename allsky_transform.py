"""
allsky_transform.py

Author: Thomas Plunkett

Date: 01/04/2026

Purpose: Refine the geometric transform (WCS) for the UTGO All Sky Camera.

"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from allsky_functions import *
from allsky_phot import *
import allsky_config as cfg
from symfit import parameters, variables, Fit, Model, sin, cos
from symfit.core.minimizers import DifferentialEvolution, LBFGSB
import numpy as np
import pandas as pd
import os
import argparse
import glob
import cv2
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord

# ------------------------------------------------------------------
# Optical distortion model
# ------------------------------------------------------------------

def calc_r(z, F, R, k3, k5):
    """
    Compute distorted radial distance in the camera projection.

    Parameters
    ----------
    z : ndarray
        Zenith angle [radians]
    F : float
        Effective focal scaling
    R : float
        Primary radial scale [pixels]
    k3, k5 : float
        Higher-order distortion coefficients

    Returns
    -------
    ndarray
        Radial distance from image center [pixels]
    """
    return R * np.sin(z / F) + k3 * z**3 + k5 * z**5


# ------------------------------------------------------------------
# Catalog cleaning
# ------------------------------------------------------------------

def drop_close_sources(df, radius_deg=1.0):
    """
    Remove stars that are too close together.

    This avoids blended sources and improves matching reliability.

    Parameters
    ----------
    df : DataFrame
        Must contain RA, DEC, Mag
    radius_deg : float
        Minimum allowed separation [degrees]

    Returns
    -------
    DataFrame
        Cleaned catalog
    """
    df = df.sort_values('Mag').reset_index(drop=True)

    coords = SkyCoord(df["RA"].values * u.deg, df["DEC"].values * u.deg)
    radius = radius_deg * u.deg

    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not keep[i]:
            continue

        sep = coords[i].separation(coords)
        close = (sep < radius) & (sep > 0 * u.deg)
        keep[close] = False

    return df[keep].reset_index(drop=True)


# ------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------

if __name__ == '__main__':

    # -----------------------------
    # Argument parsing
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Determine geometric calibration for an all-sky image"
    )
    parser.add_argument("path", type=str, help="Path to input image or video")
    parser.add_argument("output", type=str, help="Output directory")

    args = parser.parse_args()
    outpath = args.output

    # -----------------------------
    # Load / extract image
    # -----------------------------
    im_files = glob.glob(os.path.join(outpath, "frame*.jpg"))

    if len(im_files) == 0:
        get_frames(args.path, outpath)
        im_files = glob.glob(os.path.join(outpath, "frame*.jpg"))

    # Use central frame
    im = cv2.imread(im_files[len(im_files) // 2])
    if im is None:
        raise ValueError("Image could not be read")

    im_gray = convert2gray(im)
    mask_im, full_mask = create_maskim(im_gray)

    # -----------------------------
    # Time extraction
    # -----------------------------
    date_stamp, time_stamp = get_modtime(args.path)
    date_str, time_str, exp = get_imtext(im_gray, date_stamp, time_stamp)

    time_str = time_str.replace(".", ":").replace("::", ":")
    exp_time = fix_exp(exp)

    t_local = pd.to_datetime(f"{date_str} {time_str}")
    t_utc = t_local.tz_localize(cfg.TIMEZONE).tz_convert("UTC").tz_localize(None)

    astro_time = Time(t_utc)
    timestamp_str = t_utc.isoformat(timespec='seconds').replace(':', '-')

    # -----------------------------
    # Star detection
    # -----------------------------
    tas_df, _ = star_detection(im_gray, mask_im)
    tas_df = tas_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x_centroid', 'y_centroid'])
    tas_sub = tas_df.copy()

    # -----------------------------
    # Load catalog
    # -----------------------------
    ybsc_master = pd.read_csv("Yale_Cat.csv")
    ybsc_master = drop_close_sources(ybsc_master, cfg.DROP_CLOSE_RADIUS)
    ybsc_df = calc_AltAz(ybsc_master.copy(), t_utc)

    mask = (
        (ybsc_df.Alt.values >= cfg.ALT_MIN) &
        (ybsc_df.Mag.values < cfg.MAG_LIMIT)
    )

    ybsc_sub = ybsc_df.loc[mask].copy()

    # -----------------------------
    # Initial parameter guesses
    # -----------------------------
    xz_fit = cfg.xz_fit
    yz_fit = cfg.yz_fit
    F_fit = cfg.F_fit
    R_fit = cfg.R_fit
    theta_fit = cfg.theta_fit
    k3_fit = cfg.k3_fit
    k5_fit = cfg.k5_fit

    # Initial projection
    r = calc_r(np.deg2rad(ybsc_sub["Zenith"]), F_fit, R_fit, k3_fit, k5_fit)

    ybsc_sub["x_guess"] = xz_fit + r * np.cos(np.deg2rad(ybsc_sub["Az"]) - theta_fit)
    ybsc_sub["y_guess"] = yz_fit - r * np.sin(np.deg2rad(ybsc_sub["Az"]) - theta_fit)

    # -----------------------------
    # Iterative fitting loop
    # -----------------------------
    for dist in [10, 5, 2]:

        match_df = match_catalogs(ybsc_sub, tas_sub, dist)
        match_sub = match_df.dropna()

        if len(match_sub) < 10:
            print(f"Warning: low match count ({len(match_sub)})")
            continue

        # Define symfit model
        z, A, x, y = variables('z, A, x, y')
        x_z, y_z, R, F, theta, k3, k5 = parameters(
            'x_z, y_z, R, F, theta, k3, k5'
        )

        # Initial values + bounds
        x_z.value, x_z.min, x_z.max = xz_fit, xz_fit - 2, xz_fit + 2
        y_z.value, y_z.min, y_z.max = yz_fit, yz_fit - 2, yz_fit + 2
        R.value, R.min, R.max = R_fit, R_fit - 0.1, R_fit + 0.1
        F.value, F.min, F.max = F_fit, F_fit - 0.1, F_fit + 0.1
        theta.value, theta.min, theta.max = theta_fit, theta_fit - 0.1, theta_fit + 0.1

        k3.value, k3.min, k3.max = k3_fit, k3_fit - 10, k3_fit + 10
        k5.value, k5.min, k5.max = k5_fit, k5_fit - 10, k5_fit + 10

        # Model
        r_model = R * sin(z / F) + k3 * z**3 + k5 * z**5

        model = Model({
            x: x_z + r_model * cos(A - theta),
            y: y_z - r_model * sin(A - theta),
        })

        fit = Fit(
            model,
            minimizer=[DifferentialEvolution, LBFGSB],
            z=np.deg2rad(match_sub.Zenith.values),
            A=np.deg2rad(match_sub.Az.values),
            x=match_sub.x_centroid.values,
            y=match_sub.y_centroid.values,
        )

        fit_result = fit.execute()

        # Update parameters
        params = fit_result.params
        xz_fit, yz_fit = params['x_z'], params['y_z']
        F_fit, R_fit = params['F'], params['R']
        theta_fit = params['theta']
        k3_fit, k5_fit = params['k3'], params['k5']

        # Update model projection
        r = calc_r(np.deg2rad(ybsc_sub["Zenith"]), F_fit, R_fit, k3_fit, k5_fit)

        ybsc_sub["x_guess"] = xz_fit + r * np.cos(np.deg2rad(ybsc_sub["Az"]) - theta_fit)
        ybsc_sub["y_guess"] = yz_fit - r * np.sin(np.deg2rad(ybsc_sub["Az"]) - theta_fit)

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print(fit_result)

    match_df = match_catalogs(ybsc_sub, tas_df, 2)
    match_sub = match_df.dropna()

    dx = match_sub.x_centroid - match_sub.x_guess
    dy = match_sub.y_centroid - match_sub.y_guess
    alt = match_sub.Alt
    az = match_sub.Az
    res = np.sqrt(dx**2 + dy**2)

    print("Matched Stars:", len(match_sub))
    print("Median residual:", np.median(res))
    print("RMS residual:", np.sqrt(np.mean(res**2)))

    diag_name = os.path.join(outpath, f"Residuals_{timestamp_str}.jpg")

    # ------------------------------------------------------------------
    # 2D centroid plot
    fig, ax = plt.subplots()
    lim = np.max(np.abs(np.concatenate([dx, dy])))

    # Scatter
    ax.scatter(dx, dy, c='k', s=3, alpha=0.5)

    # Density
    h = ax.hist2d(dx, dy, range=[[-lim, lim], [-lim, lim]], density=True, alpha=0.25)

    # Centroid
    ax.scatter(np.median(dx), np.median(dy),
               marker='x', color='red', s=50, linewidths=2,
               label='Centroid/Offset')

    # Reference lines
    ax.axhline(0, linestyle='dashed', color='k', linewidth=1)
    ax.axvline(0, linestyle='dashed', color='k', linewidth=1)

    # Labels
    ax.set_xlabel(r'$\Delta$ X [pix]', fontsize=14)
    ax.set_ylabel(r'$\Delta$ Y [pix]', fontsize=14)

    # Equal scaling
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Grid + legend
    ax.grid(alpha=0.1)
    ax.legend()
    plt.savefig(diag_name.replace('.jpg', '_centroid.jpg'), dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # Altitude plot 
    
    fig_alt = plt.figure(figsize=(8, 6))
    gs_alt = fig_alt.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[4, 1],
        hspace=0.05, wspace=0.05
    )

    # Axes
    ax_dx_alt = fig_alt.add_subplot(gs_alt[0, 0])
    ax_dy_alt = fig_alt.add_subplot(gs_alt[1, 0], sharex=ax_dx_alt)

    # Histogram axes
    ax_dx_alt_h = fig_alt.add_subplot(gs_alt[0, 1], sharey=ax_dx_alt)
    ax_dy_alt_h = fig_alt.add_subplot(gs_alt[1, 1], sharey=ax_dy_alt)

    # Scatter
    ax_dx_alt.scatter(alt, dx, color='cornflowerblue', s=3)
    ax_dy_alt.scatter(alt, dy, color='purple', s=3)

    # Histograms
    bins = 20
    ax_dx_alt_h.hist(dx, bins=bins, orientation='horizontal', color='cornflowerblue')
    ax_dy_alt_h.hist(dy, bins=bins, orientation='horizontal', color='purple')

    # Labels
    ax_dx_alt.set_ylabel(r'$\Delta$ X [pix]', fontsize=14)
    ax_dy_alt.set_ylabel(r'$\Delta$ Y [pix]', fontsize=14)
    ax_dy_alt.set_xlabel('Altitude [deg]', fontsize=14)

    # Cleanup
    plt.setp(ax_dx_alt.get_xticklabels(), visible=False)

    for ax in [ax_dx_alt_h, ax_dy_alt_h]:
        ax.tick_params(axis='both', which='both',
                       labelbottom=False, labelleft=False)
        ax.grid(alpha=0.1)

    for ax in [ax_dx_alt, ax_dy_alt]:
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(diag_name.replace('.jpg', '_alt_residuals.jpg'), dpi=150)
    plt.show()
    
    # ------------------------------------------------------------------
    # Azimuth plot
    fig_az = plt.figure(figsize=(8, 6))
    gs_az = fig_az.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[4, 1],
        hspace=0.05, wspace=0.05
    )

    # Axes
    ax_dx_az = fig_az.add_subplot(gs_az[0, 0])
    ax_dy_az = fig_az.add_subplot(gs_az[1, 0], sharex=ax_dx_az)

    # Histogram axes
    ax_dx_az_h = fig_az.add_subplot(gs_az[0, 1], sharey=ax_dx_az)
    ax_dy_az_h = fig_az.add_subplot(gs_az[1, 1], sharey=ax_dy_az)

    # Scatter
    ax_dx_az.scatter(az, dx, color='cornflowerblue', s=3)
    ax_dy_az.scatter(az, dy, color='purple', s=3)

    # Histograms
    ax_dx_az_h.hist(dx, bins=bins, orientation='horizontal', color='cornflowerblue')
    ax_dy_az_h.hist(dy, bins=bins, orientation='horizontal', color='purple')

    # Labels
    ax_dx_az.set_ylabel(r'$\Delta$ X [pix]', fontsize=14)
    ax_dy_az.set_ylabel(r'$\Delta$ Y [pix]', fontsize=14)
    ax_dy_az.set_xlabel('Azimuth [deg]', fontsize=14)

    # Cleanup
    plt.setp(ax_dx_az.get_xticklabels(), visible=False)

    for ax in [ax_dx_az_h, ax_dy_az_h]:
        ax.tick_params(axis='both', which='both',
                       labelbottom=False, labelleft=False)
        ax.grid(alpha=0.1)

    for ax in [ax_dx_az, ax_dy_az]:
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(diag_name.replace('.jpg', '_az_residuals.jpg'), dpi=150)
    plt.show()
    
    # ------------------------------------------------------------------
    # Saving parameters
    
    param_file = os.path.join(outpath, f"AllSky_GeometricCalib_{timestamp_str}.txt")
    with open(param_file, "w") as f:
        f.write("All-Sky Camera Geometric Calibration Parameters\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Image Timestamp (UTC): {t_utc.isoformat()}\n\n")

        f.write("Fitted Parameters:\n")
        f.write(f"x_z     = {xz_fit:.6f}\n")
        f.write(f"y_z     = {yz_fit:.6f}\n")
        f.write(f"R       = {R_fit:.6f}\n")
        f.write(f"F       = {F_fit:.6f}\n")
        f.write(f"theta   = {theta_fit:.6f}\n")
        f.write(f"k3      = {k3_fit:.6e}\n")
        f.write(f"k5      = {k5_fit:.6e}\n\n")

        f.write("Fit Quality:\n")
        f.write(f"Matched stars: {len(match_sub)}\n")
        f.write(f"Median residual (pix): {np.median(res):.6f}\n")
        f.write(f"RMS residual (pix):    {np.sqrt(np.mean(res**2)):.6f}\n")

    print(f"Saved fit parameters to: {param_file}")