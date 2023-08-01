import time
start_time = time.time()

import os
import http
from lightkurve import search_tesscut
from lightkurve import DesignMatrix
from lightkurve import DesignMatrixCollection
from lightkurve import RegressionCorrector
from lightkurve import LightCurve
import numpy as np
import re
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from astroquery.skyview import SkyView
from astropy.coordinates import get_icrs_coordinates
from astropy.coordinates.name_resolve import NameResolveError
from astropy.coordinates import SkyCoord
from astropy.wcs import *
from astropy import units as u
from astropy.table import Table
import astropy.io.fits as pyfits
import csv
from astropy.io import ascii
from astropy import table
import statistics
from ztfquery import *
from ztfquery import lightcurve
import pandas as pd
from scipy.optimize import minimize

#THE FOLLOWING STATEMENTS MAY BE NEEDED IF RUNNING IN WINDOWS LINUX ENVIRONMENT:
#(NOTE: adding these may cause a Tkinter deprecation warning, but should not affect performance.)

#import matplotlib
#import tkinter
#matplotlib.use("TkAgg")

##################  TUNABLE PARAMETERS  ##########################

#For which method would you like to see detailed plots during reduction for each sector? All methods will be read out at the end.
#  1 = simple reduction using Principle Component Analysis
#  2 = simple hybrid reduction
#  3 = full hybrid reduction

primary_correction_method = 2

#Size of the TPF postage stamp to download and use for exraction and detrending.
tpf_width_height = 15

#Number of PCA Components in the Hybrid method and simple PCA correction.
additive_pca_num = 3
multiplicative_pca_num = 3
pca_only_num = 3

#Lowest DSS contour level, as fraction of peak brightness in DSS image.
#(For fields with bright stars, the default lowest level of 0.4 may be too high to see your faint source)
#This number must be less than 0.65.
lowest_dss_contour = 0.4

#Acceptable threshold for systematics in additive components:
sys_threshold = 0.2

#Maximum number of cadence-mask regions allowed:
max_masked_regions = 5 #set maximum number of regions of the light curve that can be masked out.

#Which cadence of the TESSCut file is used for the aperture selection panel
#(It is best to avoid the first or last cadences as they are often hard to see due to systematics)
plot_index = 200

#Threshold, in multiples of sigma times the median of the flux across the entire TPF,
#that divides bright from faint pixels in the calculation of principal components.
bf_threshold = 1.5

############################################
#Define function to record the positions of clicks in the pixel array image for the extraction mask.
def onclick(event):

    global ix,iy
    ix,iy = int(round(event.xdata)),int(round(event.ydata))

    global row_col_coords

    if (iy,ix) in row_col_coords:
        row_col_coords.remove((iy,ix))
        plt.plot(ix,iy,marker="x",color='red',markersize=9)
        fig.canvas.draw()

        print('removing'+str((ix,iy)))
    else:
        row_col_coords.append((iy,ix))
        plt.plot(ix,iy,marker=u"$\u2713$",color='limegreen',markersize=9)
        fig.canvas.draw()

        print('adding'+str((ix,iy)))

############################################
#Define function to record the X-positions of the cadences to mask out if needed.
def onclick_cm(event):

    global ix_cm
    ix_cm = int(round(event.xdata))

    global masked_cadence_limits
    masked_cadence_limits.append(ix_cm)
    print(ix_cm)

    plt.axvline(x=ix_cm,color='red')
    fig_cm.canvas.draw()

#############################################
#Define function for stitching the sectors together once corrected:

def lc_stitch(unstitched_lc):

    for j in range(0,len(unstitched_lc)):
        if j!=0:
            sector = str(j+1)

        lc = unstitched_lc[j]

        t = lc[:,0]
        f = lc[:,1]
        err = lc[:,2]


        if j == 0:

            full_lc_time = t
            full_lc_flux = f
            full_lc_err= err

        else:

            first_flux = np.mean(f[:10])
            last_flux = np.mean(full_lc_flux[-10:])

            scale_factor= first_flux - last_flux

            if scale_factor > 0:

                scaled_flux = f - abs(scale_factor)

            if scale_factor < 0:

                scaled_flux = f + abs(scale_factor)

            full_lc_time = np.append(full_lc_time,t)
            full_lc_flux = np.append(full_lc_flux,scaled_flux)
            full_lc_err = np.append(full_lc_err,err)

    return full_lc_time,full_lc_flux,full_lc_err

#############################################
#Define function to remove single-cadence jumps of greater or lesser than 1% of the flux on either side.
def remove_jumps(t,f,err):

    for i in range(0,1-len(t)):

        if i !=0 and i != len(f)-1 and f[i] > (0.01 * f[i-1]+f[i-1]) and f[i] > (0.01 * f[i+1] + f[i+1]):

            t = np.delete(t,i)
            f = np.delete(f,i)
            err = np.delete(err,i)

    for i in range(0,1-len(t)):

        if i !=0 and i != len(f)-1 and f[i] < (f[i-1] - 0.01 * f[i-1]) and f[i] < (f[i+1]-0.01 * f[i+1]):

            t = np.delete(t,i)
            f = np.delete(f,i)
            err = np.delete(err,i)

    return t,f,err

#function to remove spaces from a string when receiving the name of a source
def remove_spaces(string):
    new_string = ''
    for character in string:
        if character != ' ':
            new_string += character
    return new_string

#function to convert magnitudes to general flux (used for comparison of ZTF to TESS)
def mag2flux(m):
    return (10**(-m/2.5))

#function to find the indices of the closest values along the x-axis, used to find the closest observations in ZTF and TESS to scale
def find_closest_indices(array1, array2):
    i_list = []
    j_list = []
    diffs = []
    for i in range(0, len(array1)):
        for j in range(0, len(array2)):
            diffs.append(abs(array1[i] - array2[j]))
            i_list.append(i)
            j_list.append(j)
    index = diffs.index(min(diffs))
    return i_list[index], j_list[index]

#function to find the closest value in a list to a given number
def closest_val(number, data):
    diffs = list(data - number)
    index = diffs.index(min(diffs))
    return index

def objective_function(values, list1, list2):
    diff = np.abs(list1 - values * list2)
    return np.sum(diff)

def minimize_difference(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    initial_guess = 1.0
    result = minimize(objective_function, initial_guess, args=(list1, list2))
    optimal_scaling_factor = result.x[0]
    minimized_list2 = optimal_scaling_factor * list2
    return minimized_list2, 

#function to find indices of sudden spikes in the cadence-masking process:
def find_spikes(nums):
    indices = []
    for i in range(0, len(nums)):
        if (abs(nums[i][0])>= 0.1) or (abs(nums[i][1])>= 0.1) or (abs(nums[i][2])>= 0.1):
            indices.append(i)
    groups = []
    current_group = []
    for i in range(0, len(indices)):
        if i == 0 or indices[i] != indices[i-1]+1:
            current_group = [indices[i]]
        else:
            current_group.append(indices[i])
        if current_group not in groups:
            if i == len(indices)-1 or indices[i] != indices[i-1]-1:
                groups.append(current_group)
    lims = []
    for group in groups:
        if group[0] == 0:
            first_val = 0
        else:
            first_val = group[0] - 1
        if group[-1] == len(nums) - 1:
            last_val = group[-1]
        else:
            last_val = group[-1] + 1
        lims.append([first_val, last_val])
    return lims

######## BEGIN MAIN PROGRAM #################
#Define target and obtain DSS image from common name or coordinates.
try :
    target = input('Target Common Name: ')
    target_coordinates = target
    source_coordinates = get_icrs_coordinates(target)       #this requires that SIMBAD be up and working...
    print(source_coordinates)
    print("\n")

except NameResolveError:
    print("\n"+"Could not find target by name provided. Try Sky Coordinates.\n")
    print("Input as ICRS: RA,Dec  (in Decimal Degrees, with no space)")

    input_coord_string = input('RA,Dec: ')
    input_coord_split = re.split("\s|[,]|[,\s]",input_coord_string)

    ra = float(input_coord_split[0])
    dec = float(input_coord_split[1])

    source_coordinates = SkyCoord(ra,dec,frame='icrs',unit='deg')

    target = input('Desired object name for output files: ')
    target_coordinates = str(ra)+" "+str(dec)

    print(source_coordinates)
    print("\n")

# try:
#     input_coord_string = input('RA,Dec: ')
#     input_coord_split = re.split("\s|[,]|[,\s]",input_coord_string)
#     ra = float(input_coord_split[0])
#     dec = float(input_coord_split[1])
#     source_coordinates = SkyCoord(ra,dec,frame='icrs',unit='deg')
#     target = input('Desired object name for output files: ')
#     target_coordinates = str(ra)+" "+str(dec)
#     query = lightcurve.LCQuery.query_position(ra=ra, dec=dec, radius_arcsec = 5, bandname = "r", bad_catflags_mask = 65535) #queries ZTF data
#     data = query.data
#     oid = statistics.mode(data['oid']) #have data in 5" radius, this makes sure we only get the main source's data
#     ztf_df = data.loc[data['oid'] == oid]
#     ztf_df.reset_index(inplace=True, drop=True)
#     print("Mean magnitude is", str(np.mean(ztf_df['mag']))) 

# except NameResolveError:
#     target = input('Target Common Name: ')
#     source_coordinates = get_icrs_coordinates(target) #this requires that SIMBAD be up and working...
#     ra = source_coordinates.ra.degree
#     dec = source_coordinates.dec.degree
#     query = lightcurve.LCQuery.query_position(ra=ra, dec=dec, radius_arcsec = 5, bandname = "r", bad_catflags_mask = 65535) #32768)
#     data = query.data
#     oid = statistics.mode(data['oid'])
#     ztf_df = data.loc[data['oid'] == oid]
#     ztf_df.reset_index(inplace=True, drop=True)
#     print("Mean magnitude for", str(np.mean(ztf_df['mag'])))

dss_image = SkyView.get_images(position=source_coordinates,survey='DSS',pixels=str(400))
wcs_dss = WCS(dss_image[0][0].header)
dss_pixmin = np.min(dss_image[0][0].data)
dss_pixmax = np.max(dss_image[0][0].data)
dss_pixmean = np.mean(dss_image[0][0].data)

dss_head = dss_image[0][0].header
dss_ra = dss_head['CRVAL1']
dss_dec = dss_head['CRVAL2']

#Retrieve the available tesscut data for FFI-only targets.
sector_data = search_tesscut(target_coordinates)
num_obs_sectors = len(sector_data)

if num_obs_sectors == 0:
    print("This object has not been observed by TESS.")
    sys.exit()

#If object is observed by TESS and specified Cycle makes sense, begin aperture
#selection and extraction!

for i in range(0,len(sector_data)):
    try:
        tpf = sector_data[i].download(cutout_size=(tpf_width_height, tpf_width_height)) #gets earliest sector

        sector_number = tpf.get_header()['SECTOR']
        sec = str(sector_number)
        ccd = tpf.get_header()['CCD']
        cam = tpf.get_header()['CAMERA']

        print("Generating pixel map for sector "+sec+".\n")

        #Check that this object is actually on silicon and getting data (not always the case just because TESSCut says so).
        #By making a light curve from a dummy aperture of the middle 5x5 square and seeing if its mean flux is zero.

        aper_dummy = np.zeros(tpf[0].shape[1:], dtype=bool) #blank
        aper_dummy[int(tpf_width_height/2-3):int(tpf_width_height/2+3),int(tpf_width_height/2-3):int(tpf_width_height/2+3)] = True
        lc_dummy = tpf.to_lightcurve(aperture_mask=aper_dummy)

        if np.mean(lc_dummy.flux) == 0:
            print("This object is not actually on silicon.")
            sys.ext()

        else:

            hdu = tpf.get_header(ext=2)

            #Get WCS information and flux stats of the TPF image.
            tpf_wcs = WCS(tpf.get_header(ext=2))

            pixmin = np.min(tpf.flux[plot_index]).value
            pixmax = np.max(tpf.flux[plot_index]).value
            pixmean = np.mean(tpf.flux[plot_index]).value

            temp_min = float(pixmin)
            # print(temp_min)
            temp_max = float(1e-3*pixmax+pixmean)
            #temp_max = pixmax
            # print(temp_max)

            #Create a blank boolean array for the aperture, which will turn to TRUE when pixels are selected.

            aper = np.zeros(tpf[0].shape[1:], dtype=bool) #blank
            aper_mod = aper.copy()       #For the source aperture
            aper_buffer = aper.copy()    #For the source aperture plus a buffer region to exclude from both additive and mult. regressors

            aper_width = tpf[0].shape[1]
            #Plot the TPF image and the DSS contours together, to help with aperture selection, along with the starter aperture.

            if lowest_dss_contour == 0.4:
                dss_levels = [0.4*dss_pixmax,0.5*dss_pixmax,0.75*dss_pixmax]
            elif lowest_dss_contour < 0.4:
                dss_levels = [lowest_dss_contour*dss_pixmax,0.4*dss_pixmax,0.5*dss_pixmax,0.75*dss_pixmax]
            elif lowest_dss_contour > 0.4:
                dss_levels = [lowest_dss_contour*dss_pixmax,0.65*dss_pixmax,0.85*dss_pixmax]

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection=tpf_wcs)
            # ax.imshow(tpf.flux[200],vmin=pixmin,vmax=1e-3*pixmax+pixmean)
            ax.imshow(tpf.flux[plot_index].value,vmin=temp_min,vmax=temp_max)
            ax.contour(dss_image[0][0].data,transform=ax.get_transform(wcs_dss),levels=dss_levels,colors='white',alpha=0.9)
            ax.scatter(aper_width/2.0,aper_width/2.0,marker='x',color='k',s=8)

            ax.set_xlim(-0.5,aper_width-0.5)  #This section is needed to fix the stupid plotting issue in Python 3.
            ax.set_ylim(-0.5,aper_width-0.5)

            plt.title('Define extraction pixels:')
            row_col_coords = []
            cid = fig.canvas.mpl_connect('button_press_event',onclick)

            plt.show()
            plt.close(fig)

            fig.canvas.mpl_disconnect(cid)

            buffer_pixels = []      #Define the buffer pixel region.

            if len(row_col_coords) == 0:
                print('No mask selected; skipping this Sector.')

            else:

                for i in range(0,len(row_col_coords)):

                    aper_mod[row_col_coords[i]] = True

                    row_same_up_column = (row_col_coords[i][0],row_col_coords[i][1]+1)
                    row_same_down_column = (row_col_coords[i][0],row_col_coords[i][1]-1)
                    column_same_down_row = (row_col_coords[i][0]-1,row_col_coords[i][1])
                    column_same_up_row = (row_col_coords[i][0]+1,row_col_coords[i][1])

                    bottom_left_corner = (row_col_coords[i][0]-1,row_col_coords[i][1]-1)
                    top_right_corner = (row_col_coords[i][0]+1,row_col_coords[i][1]+1)
                    top_left_corner = (row_col_coords[i][0]+1,row_col_coords[i][1]-1)
                    bottom_right_corner = (row_col_coords[i][0]-1,row_col_coords[i][1]+1)

                    buffer_line = (row_same_up_column,row_same_down_column,column_same_up_row,column_same_down_row,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner)
                    buffer_pixels.append(buffer_line)

                    for coord_set in buffer_line:
                            aper_buffer[coord_set[0],coord_set[1]]=True


                #Create a mask that finds all of the bright, source-containing regions of the TPF.
                allbright_mask = tpf.create_threshold_mask(threshold=bf_threshold,reference_pixel=None)
                allfaint_mask = ~allbright_mask

                allbright_mask &= ~aper_buffer
                allfaint_mask &= ~aper_buffer

                #Remove any empty flux arrays from the downloaded TPF before we even get started:

                boolean_orignans = []

                for i in range(0,len(tpf.flux)):

                    if np.sum(tpf.flux[i] == 0) or np.isnan(np.sum(tpf.flux[i])) == True:

                        nanflag = True

                    else:

                        nanflag = False

                    boolean_orignans.append(nanflag)

                boolean_orignans_array = np.array(boolean_orignans)
                tpf = tpf[~boolean_orignans_array]

                #Get the additive background first:

                additive_hybrid_pcas = additive_pca_num

                additive_bkg = DesignMatrix(tpf.flux[:, allfaint_mask]).pca(additive_hybrid_pcas)
                additive_bkg_and_constant = additive_bkg.append_constant()

                #Add a module to catch possible major systematics that need to be masked out before continuing:          
                limits = find_spikes(additive_bkg.values)
                while len(limits) > 0:
                    limits = find_spikes(additive_bkg.values)
                    print(limits)
                    cadence_mask = ~(((tpf.time.value >= tpf.time[limits[0][0]].value) & (tpf.time.value <= tpf.time[limits[0][1]].value)))
                    tpf = tpf[cadence_mask]
                    additive_bkg = DesignMatrix(tpf.flux[:,allfaint_mask]).pca(additive_hybrid_pcas)
                    additive_bkg_and_constant = additive_bkg.append_constant()
                    limits = find_spikes(additive_bkg.values)
                    
                r = RegressionCorrector(LightCurve(time=tpf.time, flux=tpf.time.value*0))

                corrected_pixels = []
                for idx in range(allbright_mask.sum()):
                    r.lc.flux = tpf.flux[:, allbright_mask][:, idx]
                    r.lc.flux_err = tpf.flux_err[:, allbright_mask][:, idx]
                    r.correct(additive_bkg_and_constant)
                    corrected_pixels.append(r.corrected_lc.flux)

                #Getting the multiplicative effects now from the bright pixels.
                multiplicative_hybrid_pcas = multiplicative_pca_num
                multiplicative_bkg = DesignMatrix(np.asarray(corrected_pixels).T).pca(multiplicative_hybrid_pcas)

                #Create a design matrix using only the multiplicative components determined from the additively-corrected bright sources for simple hybrid method:
                dm_mult = multiplicative_bkg
                dm_mult = dm_mult.append_constant()

                #Now get the raw light curve.
                lc = tpf.to_lightcurve(aperture_mask=aper_mod)
            #  lc = lc[lc.flux_err > 0]        #This was suggested by an error message to prevent the "flux uncertainties" problem.

                median_flux_precorr = np.median(lc.flux.value) #Calculate the median flux before the background subtraction upcoming.

                #Begin the SIMPLE HYBRID METHOD
                #First, simple background subtraction to handle additive effects:
                lc_bg = tpf.to_lightcurve(method='sap',corrector=None,aperture_mask = allfaint_mask)

                num_pixels_faint = np.count_nonzero(allfaint_mask)
                num_pixels_mask = np.count_nonzero(aper_mod)
                percent_of_bg_in_src = num_pixels_mask / num_pixels_faint

                lc_bg_time = lc_bg.time.value
                lc_bg_flux = lc_bg.flux.value
                lc_bg_fluxerr = lc_bg.flux_err.value

                lc_bg_scaled = lc_bg_flux - (1-percent_of_bg_in_src)*lc_bg_flux

                lc.flux = lc.flux.value - lc_bg_scaled

                #Replace any errors that are zero or negative with the mean error:

                mean_error = np.mean(lc.flux_err[np.isfinite(lc.flux_err)])
                lc.flux_err = np.where(lc.flux_err == 0,mean_error,lc.flux_err)
                lc.flux_err = np.where(lc.flux_err < 0,mean_error,lc.flux_err)
                lc.flux_err = lc.flux_err.value

                #And correct regressively for the multiplicative effects in the simple hybrid method:

                corrector_1 = RegressionCorrector(lc)
                clc = corrector_1.correct(dm_mult)

                #The background subtraction can sometimes cause fluxes below the source's median
                #to be slightly negative; this enforces a minimum of zero, but can be ignored.

                if np.min(clc.flux.value) < 0:

                    dist_to_zero = np.abs(np.min(clc.flux.value))
                    clc.flux = clc.flux.value + dist_to_zero

                # Optional additive correction back to original median:
                median_flux_postsub = np.median(clc.flux.value)
                additive_rescale_factor = median_flux_precorr - median_flux_postsub
                # clc.flux = clc.flux.value + additive_rescale_factor    #uncomment if you want to use this.

                var_amplitude = np.max(clc.flux.value) - np.min(clc.flux.value)
                percent_variability = (var_amplitude / median_flux_precorr)*100



                #For the FULL HYBRID METHOD:
                #We make a fancy hybrid design matrix collection that has two orders of the additive effects, and the multiplicative effects.

                additive_bkg_squared = deepcopy(additive_bkg)
                additive_bkg_squared.df = additive_bkg_squared.df**2

                dmc = DesignMatrixCollection([additive_bkg_and_constant, additive_bkg_squared, multiplicative_bkg])
                lc_full = tpf.to_lightcurve(aperture_mask=aper_mod)
                clc_full = RegressionCorrector(lc_full).correct(dmc)

                #Now we begin the SIMPLE PCA METHOD with components of all non-source pixels.

                raw_lc_OF = tpf.to_lightcurve(aperture_mask=aper_mod)

                #Replace any errors that are zero or negative with the mean error:
                raw_lc_OF.flux_err = np.where(raw_lc_OF.flux_err == 0,mean_error,raw_lc_OF.flux_err)
                raw_lc_OF.flux_err = np.where(raw_lc_OF.flux_err < 0,mean_error,raw_lc_OF.flux_err)
                raw_lc_OF.flux_err = np.where(np.isnan(raw_lc_OF.flux_err)==True,mean_error,raw_lc_OF.flux_err)

            #    raw_lc_OF = raw_lc_OF[raw_lc_OF.flux_err > 0]   #This was suggested by an error message to prevent the "flux uncertainties" problem.
                regressors_OF = tpf.flux[:,~aper_mod]

                number_of_pcas = pca_only_num

                dm_OF = DesignMatrix(regressors_OF,name='regressors')
                dm_pca_OF = dm_OF.pca(pca_only_num)
                dm_pca_OF = dm_pca_OF.append_constant()

                corrector_pca_OF = RegressionCorrector(raw_lc_OF)
                corrected_lc_pca_OF = corrector_pca_OF.correct(dm_pca_OF)

                #AND PLOT THE CORRECTED LIGHT CURVE.

                fig2 = plt.figure(figsize=(12,8))
                gs = gridspec.GridSpec(ncols=3, nrows=3,wspace=0.5,hspace=0.5,width_ratios=[1,1,2])
                f_ax1 = fig2.add_subplot(gs[0, :])
                f_ax1.set_title(target+': Corrected Light Curve')
                f_ax2 = fig2.add_subplot(gs[1, :-1])
                if primary_correction_method == 1:
                    f_ax2.set_title('Principal Components')
                    f_ax4 = fig2.add_subplot(gs[1:,-1])
                elif primary_correction_method == 2 or primary_correction_method == 3:
                    f_ax2.set_title('Additive Components')
                    f_ax3 = fig2.add_subplot(gs[2:,:-1])
                    f_ax3.set_title('Multiplicative Components')
                    f_ax4 = fig2.add_subplot(gs[1:,-1])


                #if primary_correction_method == 1:
                corrected_lc_pca_OF.plot(ax=f_ax1, label='PCA')

                #elif primary_correction_method == 2:
                clc.plot(ax=f_ax1, label='Simple Hybrid')

                #elif primary_correction_method == 3:
                clc_full.plot(ax=f_ax1, label='Full hybrid')

                f_ax1.legend(loc='best')

                #if primary_correction_method == 1:
                    #f_ax2.plot(raw_lc_OF.time.value,dm_pca_OF.values[:,0:-1])

                #elif primary_correction_method == 2 or primary_correction_method == 3:
                f_ax2.plot(raw_lc_OF.time.value,additive_bkg.values)
                f_ax3.plot(raw_lc_OF.time.value,multiplicative_bkg.values + np.arange(multiplicative_bkg.values.shape[1]) * 0.3)

                tpf.plot(ax=f_ax4,aperture_mask=aper_mod,title='Aperture')


###############################################################################
                    ## This section creates individual directories for each object in which the Quaver-processed light curve data is stored
                    ## and saves the output images and light curves to that directory. The output files WILL be overwritten with each Quaver run.
##############################################################################
                directory = str(target).replace(" ","")
                target_safename = target.replace(" ","")
                try:
                    os.makedirs('quaver_output/'+target_safename)
                    print("Directory '% s' created\n" % directory)
                    if primary_correction_method == 1:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_SimplePCA_sector'+sec+'.pdf',format='pdf')
                        plt.show()
                    elif primary_correction_method == 2:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_SimpleHybrid_sector'+sec+'.pdf',format='pdf')
                        plt.show()
                    elif primary_correction_method == 3:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_FullHybrid_sector'+sec+'.pdf',format='pdf')
                        plt.show()

                except FileExistsError:
                    print("Saving to folder '% s'\n" % directory)
                    if primary_correction_method == 1:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_SimplePCA_sector'+sec+'.pdf',format='pdf')
                        plt.show()
                    elif primary_correction_method == 2:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_SimpleHybrid_sector'+sec+'.pdf',format='pdf')
                        plt.show()
                    elif primary_correction_method == 3:
                        plt.savefig('quaver_output/'+target_safename+'/'+target_safename+'_FullHybrid_sector'+sec+'.pdf',format='pdf')
                        plt.show()

                #Create saveable formats for the light curves and save to directory:
                pca_time = [time + 56999.5 for time in corrected_lc_pca_OF.time.value]
                pca_flux = [flux - additive_rescale_factor for flux in corrected_lc_pca_OF.flux.value]
                pca_err = corrected_lc_pca_OF.flux_err.value

                simple_hybrid_time = [time + 56999.5 for time in clc.time.value]
                simple_hybrid_flux = [flux - additive_rescale_factor for flux in clc.flux.value]
                simple_hybrid_err = clc.flux_err.value

                full_hybrid_time = [time + 56999.5 for time in clc_full.time.value]
                full_hybrid_flux = [flux - additive_rescale_factor for flux in clc_full.flux.value]
                full_hybrid_err = clc_full.flux_err.value

                tess_times, tess_fluxes, tess_errs = full_hybrid_time, full_hybrid_flux, full_hybrid_err

                ztf_time_tot, ztf_flux_tot = np.array(ztf_df['hjd'] - 2400000.5), np.array(mag2flux(ztf_df['mag']))
                ztf_time, ztf_flux = [], []
                for i in range(0, len(ztf_time_tot)):
                    if ztf_time_tot[i] >= min(tess_times) and ztf_time_tot[i] <= max(tess_times):
                        ztf_time.append(ztf_time_tot[i])
                        ztf_flux.append(ztf_flux_tot[i])
                print("Number of ZTF observations=", str(len(ztf_time)))
                
                tess_ztf_times = []
                tess_ztf_fluxes = []
                ztf_times_for_scaling = []
                ztf_fluxes_for_scaling = []
                for i in range(0, len(ztf_time)):
                    idx = closest_val(ztf_time[i], tess_times)
                    if abs(ztf_time[i] - tess_times[idx]) <= 0.5:
                        tess_ztf_times.append(tess_times[idx])
                        tess_ztf_fluxes.append(tess_fluxes[idx])
                        ztf_times_for_scaling.append(ztf_time[i])
                        ztf_fluxes_for_scaling.append(ztf_flux[i])

                scaled_ztf_fluxes, scaling_factor = minimize_difference(tess_ztf_fluxes, ztf_fluxes_for_scaling)
                print("Optimal Scaling Factor:", scaling_factor)

                if len(ztf_times_for_scaling) != 0:
                    residual_percentages = []
                    for i in range(0, len(ztf_times_for_scaling)):
                        index = closest_val(ztf_times_for_scaling[i], tess_ztf_times[i])
                        tess_time, tess_flux = tess_ztf_times[index], tess_ztf_fluxes[index]
                        residual_percent.append(100*abs(1-(ztf_fluxes_for_scaling[i]/tess_flux)))

                    
                    num = 0
                    for val in residual_percentages:
                        if val > 25:
                            num += 1
                    print("Fraction of residual percentages over 25%: ", num)

                    # if len(residual_percentages) != 0 and (num/len(residual_percentages)) <= 0.5:
                    plt.figure(figsize=(14, 5))
                    plt.plot(full_hybrid_time, full_hybrid_flux, zorder=0, color='blue', label='TESS')
                    plt.scatter(ztf_time, ztf_flux, color='orange', s=10, zorder=2, label='ZTF')
                    plt.plot(ztf_time, ztf_flux, color='orange', zorder=1)
                    plt.xlim(min(full_hybrid_time), max(full_hybrid_time))
                    plt.ylim(0.9*min(full_hybrid_flux), 1.1*max(full_hybrid_flux))
                    plt.show()
                    # else:
                    #     print("This sector doesn't match ZTF observations.")                
                else: 
                    print("There are no ZTF observations during this sector.")
                print("\nMoving to next sector.\n")

            
    # If target coordinates are too close to edge on approach, this will skip that sector and read the next.
    # If target coordinates are too close to edge on exit, this will skip that sector.
    ## WARNING: May also occur if connection to HEASARC could not be made. Check website and/or internet connection.

    except (http.client.IncompleteRead):

        print("Unable to download FFI cutout. Desired target coordinates may be too near the edge of the FFI.\n")
        print("Could be inability to connect to HEASARC. Check website availability and/or internet connection.\n")

        if i != num_obs_sectors-1:

          print("\nMoving to next sector.\n")

        continue

print ("Done!")

print("--- %s seconds ---" % (time.time() - start_time))