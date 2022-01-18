# Copyright (C) President and Fellows of Harvard College 2021

# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public
#   License along with this program.  If not, see
#   <http://www.gnu.org/licenses/>.

# The EXPOS model uses a digital elevation model to estimate exposed and
# protected areas for a given wind direction and inflection angle.

# The input file is assumed to be a raster file in GeoTiff format with 
# missing values represented by zero.  Cells may be rectangular but 
# horizontal and vertical units must be the same. Columns are assumed
# to be closely aligned with true north (if not, wind direction values
# must be adjusted accordingly). The name of the input file is 
# assumed to be "dem.tif".

# The output file is a raster file in GeoTiff format with the following
# values: 0 = missing data, 1 = protected, 2 = exposed. Output files
# are named "expos-xxx-yy.tif" where xxx is the wind direction and yy
# is the inflection angle.

# Emery R. Boose
# January 2022

# Python version 3.7.11

# Note: If setting PROJ_LIB as an environmental variable at the OS level
# causes problems with other programs, try setting it as below substituting
# the correct path for your system.

### MODULES ###############################################

import os

# set PROJ_LIB as Python environmental variable
os.environ['PROJ_LIB'] = "C:/Anaconda3/envs/hf/Library/share/proj"

import sys
import math
import fiona
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.plot import show
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from descartes import PolygonPatch


### INTERNAL FUNCTIONS ####################################

# check_file_exists displays an error message and stops execution if
# the specified file does not exist.
#   file_name - name of file
# no return value

def check_file_exists(file_name):
    if os.path.exists(file_name) == False or os.path.isfile(file_name) == False:
        sys.exit("File not found: " + file_name)

# get_subdirectory returns the subdirectory for a given filename
# and stops execution if the filename is not supported.
#   filename - name of file
# returns subdirectory

def get_subdirectory(filename):
    if filename == "dem":
        subdir = "/dem/"

    elif "expos" in filename:
        subdir = "/exposure/"

    elif "damage" in filename:
        subdir = "/damage/"

    else:
        sys.exit("File name not supported")

    return subdir

# get_row_order returns True if the row order remains unchanged 
# (quadrants I-II) or False if the row order is reversed
# (quadrants III-IV).
#   wind_direction - wind direction (degrees)
# returns row order (True or False)

def get_row_order(wind_direction):
    if wind_direction > 90 and wind_direction < 270:
        row_order = False
    
    else:
        row_order = True

    return row_order

# get_col_order returns True if the column order remains unchanged
# (quadrants I, IV) or False if the column order is reversed
# (quadrants II-III).
#   wind_direction - wind direction (degrees)
# returns column order (True or False)

def get_col_order(wind_direction):
    if wind_direction > 180 and wind_direction < 360:
        row_order = True
    
    else:
        row_order = False

    return row_order

# get_transposed_wind_direction returns the transposed wind direction
# after the actual wind direction is shifted to quadrant II.
#   wdir - wind direction (degrees)
# returns the transposed wind direction (degrees)

def get_transposed_wind_direction(wdir):
    # quadrant I
    if wdir >= 0 and wdir <= 90:
        t_dir = 360 - wdir

    # quadrant IV
    elif wdir > 90 and wdir < 180:
        t_dir = wdir + 180
    # quadrant III
    elif wdir >= 180 and wdir < 270:
        t_dir = 540 - wdir;

    # quadrant II
    elif wdir >= 270 and wdir <= 360:
        t_dir = wdir;

    return t_dir

# west_north_west creates and saves a raster of exposure values for
# transposed wind directions between 270 degrees and the cell diagonal 
# (WNW). The transposed matrix of elevation values is processed in column
# major order.
#   wind_direction - wind direction (degrees)
#   inflection_angle - inflection angle (degrees)
#   t_dir - transposed wind direction (degrees)
#   save - whether to save results to file
#   console - whether to display messages in console
# returns a raster of modeled exposure values

def west_north_west(wind_direction, inflection_angle, t_dir, save, console):
    # get current working directory
    cwd = os.getcwd()
 
    # read dem file in GeoTiff format
    dem_path = cwd + "/dem/dem.tif"
    check_file_exists(dem_path)
    dem_r = rio.open(dem_path)
 
    # get profile
    profile = dem_r.profile
    profile.update(dtype='int32', nodata=-9999, count=1)

    # get number of rows & columns
    nrows = dem_r.height
    ncols = dem_r.width

    # get extent
    xmn = dem_r.bounds.left
    xmx = dem_r.bounds.right
    ymn = dem_r.bounds.bottom
    ymx = dem_r.bounds.top
    
    # get cell dimensions
    cell_x = (xmx-xmn)/ncols
    cell_y = (ymx-ymn)/nrows
  
    # set exposure values
    pro_value = 1
    exp_value = 2

    # get row & column order
    row_order = get_row_order(wind_direction)
    col_order = get_col_order(wind_direction)

    # create array
    aa = dem_r.read(1)
    dem_r.close()

    # flip array as needed
    if row_order == True and col_order == True:
        dem_a = aa

    elif row_order == False and col_order == True:
        dem_a = np.flip(aa, 0)

    elif row_order == True and col_order == False:
        dem_a = np.flip(aa, 1)

    elif row_order == False and col_order == False:
        xx = np.flip(aa, 0)
        dem_a = np.flip(xx, 1)
  
    # create exposure array
    expos_a = np.zeros((nrows, ncols), dtype=np.int32)

    # create vectors for intermediate values
    p_shift = [None] * ncols  # shift value for each column
    h_pos   = [None] * nrows  # height of land column number
    h_elev  = [None] * nrows  # height of land elevation
  
    # get tangent of inflection angle
    tan_inf = math.tan(inflection_angle*math.pi/180)

    # get adjustment for transposed wind direction
    adj = math.tan((t_dir - 270)*math.pi/180)
  
    # get shift value for each column
    for j in range(0, ncols):
        p_shift[j] = round(adj*j*cell_x/cell_y)

    # calculate exposure values
    for j in range(0, ncols):
        # display every 10th col number
        if j % 10 == 0:
            print("              ", end="")
            print("\rcol", j, end="")

        # first column exposed by default
        if j == 0:
            for i in range(0, nrows):
                h_pos[i] = 0
                h_elev[i] = dem_a[i, 0]
                expos_a[i, 0] = exp_value

        else:
            # shift for current column (0 or 1)
            shift = p_shift[j] - p_shift[j-1]
 
            # shift by one row
            if shift == 1:
                for i in range((nrows-2), 0):
                    h_pos[i+1] = h_pos[i]
                    h_elev[i+1] = h_elev[i]

                h_pos[0] = j
                h_elev[0] = dem_a[0, j]
                expos_a[i, j] = exp_value

            for i in range((shift+1), nrows):
                # exposed (higher elevation)
                if dem_a[i, j] >= h_elev[i]:
                    h_pos[i] = j
                    h_elev[i] = dem_a[i, j]
                    expos_a[i, j] = exp_value
        
                else:
                    x_dist = (p_shift[j] - p_shift[h_pos[i]])*cell_x
                    y_dist = (j - h_pos[i])*cell_y
                    xy_dist = (x_dist**2 + y_dist**2)**0.5
                    z_dist = xy_dist * tan_inf
          
                    # exposed (beyond wind shadow)
                    if dem_a[i, j] >= h_elev[i] - z_dist:
                        h_pos[i] = j
                        h_elev[i] = dem_a[i, j]
                        expos_a[i, j] = exp_value
        
                    # protected (in wind shadow)
                    else:
                        expos_a[i, j] = pro_value

    # set missing values to zero
    mask = np.where(dem_a != 0, 1, 0)
    zz = np.multiply(expos_a, mask)

    # flip array as needed
    if row_order == True and col_order == True:
        expos_f = zz

    elif row_order == False and col_order == True:
        expos_f = np.flip(zz, 0)

    elif row_order == True and col_order == False:
        expos_f = np.flip(zz, 1)

    elif row_order == False and col_order == False:
        xx  = np.flip(zz, 0)
        expos_f = np.flip(xx, 1)

    # output
    if save == True:
        # save modeled values in a Geotiff file
        expos_file = cwd + "/exposure/expos-" + str(wind_direction).zfill(3) + "-" + str(inflection_angle).zfill(2) + ".tif"
    
        expos_tif = rio.open(expos_file, 'w', **profile)
        expos_tif.write(expos_f, 1)
        expos_tif.close()
   
        if console == True:
          print("\nSaving to", expos_file, "\n")
  
    else:
        # return modeled values as raster
        return expos_f

# north_north_west creates and saves a raster of exposure values for
# transposed wind directions between the cell diagonal and 360 degrees 
# (NNW). The transposed matrix of elevation values is processed in row
# major order.
#   wind_direction - wind direction (degrees)
#   inflection_angle - inflection angle (degrees)
#   t_dir - transposed wind direction (degrees)
#   save - whether to save results to file
#   console - whether to display messages in console
# returns a raster of modeled exposure values

def north_north_west(wind_direction, inflection_angle, t_dir, save, console):
    # get current working directory
    cwd = os.getcwd()
 
    # read dem file in GeoTiff format
    dem_path = cwd + "/dem/dem.tif"
    check_file_exists(dem_path)
    dem_r = rio.open(dem_path)

    # get profile
    profile = dem_r.profile
    profile.update(dtype='int32', nodata=-9999, count=1)
 
    # get number of rows & columns
    nrows = dem_r.height
    ncols = dem_r.width

    # get extent
    xmn = dem_r.bounds.left
    xmx = dem_r.bounds.right
    ymn = dem_r.bounds.bottom
    ymx = dem_r.bounds.top
    
    # get cell dimensions
    cell_x = (xmx-xmn)/ncols
    cell_y = (ymx-ymn)/nrows
  
    # set exposure values
    pro_value = 1
    exp_value = 2

    # get row & column order
    row_order = get_row_order(wind_direction)
    col_order = get_col_order(wind_direction)

    # create array
    aa = dem_r.read(1)
    dem_r.close()

    # flip array as needed
    if row_order == True and col_order == True:
        dem_a = aa

    elif row_order == False and col_order == True:
        dem_a = np.flip(aa, 0)

    elif row_order == True and col_order == False:
        dem_a = np.flip(aa, 1)

    elif row_order == False and col_order == False:
        xx = np.flip(aa, 0)
        dem_a = np.flip(xx, 1)
  
    # create exposure array
    expos_a = np.zeros((nrows, ncols), dtype=np.int32)

    # create vectors for intermediate values
    p_shift = [None] * nrows  # shift value for each row
    h_pos   = [None] * ncols  # height of land row number
    h_elev  = [None] * ncols  # height of land elevation
  
    # get tangent of inflection angle
    tan_inf = math.tan(inflection_angle*math.pi/180)

    # get adjustment for transposed wind direction
    adj = math.tan((360 - t_dir)*math.pi/180)
  
    # get shift value for each row
    for i in range(0, nrows):
        p_shift[i] = round(adj*(i-1)*cell_y/cell_x)

    # calculate exposure values
    for i in range(0, nrows):
        # display every 10th row number
        if i % 10 == 0:
            print("              ", end="")
            print("\rrow", i, end="")

        # first row is exposed by default
        if i == 0:
            for j in range(0, ncols):
                h_pos[j] = 0
                h_elev[j] = dem_a[i, j]
                expos_a[i, j] = exp_value

        else:
            # shift for current row (0 or 1)
            shift = p_shift[i] - p_shift[i-1];

            # shift by one column
            if shift == 1:
                for j in range((ncols-1), 0):
                    h_pos[j+1] = h_pos[j]
                    h_elev[j+1] = h_elev[j]
      
                h_pos[0] = i
                h_elev[0] = dem_a[i, 0]
                expos_a[i, j] = exp_value

            for j in range((shift+1), ncols):
                # exposed (higher elevation)
                if dem_a[i, j] >= h_elev[j]:
                    h_pos[j] = i
                    h_elev[j] = dem_a[i, j]
                    expos_a[i, j] = exp_value
        
                else:
                    x_dist = (p_shift[i] - p_shift[h_pos[j]])*cell_x
                    y_dist = (i - h_pos[j])*cell_y
                    xy_dist = (x_dist**2 + y_dist**2)**0.5
                    z_dist = xy_dist * tan_inf
          
                    # exposed (beyond wind shadow)
                    if dem_a[i, j] >= h_elev[j] - z_dist:
                        h_pos[j] = i
                        h_elev[j] = dem_a[i, j]
                        expos_a[i, j] = exp_value
          
                    # protected (in wind shadow)
                    else:
                        expos_a[i, j] = pro_value

    # set missing values to zero
    mask = np.where(dem_a != 0, 1, 0)
    zz = np.multiply(expos_a, mask)
  
    # flip array as needed
    if row_order == True and col_order == True:
        expos_f = zz

    elif row_order == False and col_order == True:
        expos_f = np.flip(zz, 0)

    elif row_order ==True and col_order == False:
        expos_f = np.flip(zz, 1)

    elif row_order == False and col_order == False:
        xx  = np.flip(zz, 0)
        expos_f = np.flip(xx, 1)

    # output
    if save == True:
        # save modeled values in a Geotiff file
        expos_file = cwd + "/exposure/expos-" + str(wind_direction).zfill(3) + "-" + str(inflection_angle).zfill(2) + ".tif"
    
        expos_tif = rio.open(expos_file, 'w', **profile)
        expos_tif.write(expos_f, 1)
        expos_tif.close()
   
        if console == True:
          print("\nSaving to", expos_file, "\n")
  
    else:
        # return modeled values as raster
        return expos_f


### UTILITY FUNCTIONS #####################################

# expos_set_path sets the path for the current set of model runs.
#   exp_path - path for current model runs
#   console - whether to display messages in console
# no return value

def expos_set_path(exp_path, console=True):
    if exp_path == "":
        sys.exit("Need to enter a path")

    elif os.path.exists(exp_path) == False:
        sys.exit("Path does not exist")

    os.chdir(exp_path)

    if console == True:
        print("Path set to", exp_path)


### MODELING FUNCTIONS ####################################

# expos_model uses a raster file of elevation values, a specified wind
# direction, and a specified inflection angle to create a raster file
# of wind exposure values (0 = missing data, 1 = protected, 2 = exposed).
#   wind_direction - wind direction (degrees)
#   inflection_angle - inflection angle (degrees)
#   save - whether to save results to file
#   console - whether to display messages in console
# no return value

def expos_model(wind_direction, inflection_angle, save=True, console=True):
    # get current working directory
    cwd = os.getcwd()
 
    # check wind direction
    if wind_direction < 0 or wind_direction > 360:
        sys.exit("Please supply wind direction in range 0-360 degrees")

    # check inflection angle
    if inflection_angle < 0 or inflection_angle > 90:
        sys.exit("Please supply inflection angle in range 0-90 degrees")

    # read dem file in GeoTiff format
    dem_path = cwd + "/dem/dem.tif"
    check_file_exists(dem_path)
    dem_r = rio.open(dem_path)
 
    # get number of rows & columns
    nrows = dem_r.height
    ncols = dem_r.width

    # get extent
    xmn = dem_r.bounds.left
    xmx = dem_r.bounds.right
    ymn = dem_r.bounds.bottom
    ymx = dem_r.bounds.top
    
    dem_r.close()

    # get cell dimensions
    cell_x = (xmx-xmn)/ncols
    cell_y = (ymx-ymn)/nrows
  
    # get angle of cell diagonal
    cell_diagonal = 360 - 180*math.atan(cell_x/cell_y)/math.pi;

    # get transposed wind direction
    t_dir = get_transposed_wind_direction(wind_direction)
  
    # create exposure map
    if t_dir < cell_diagonal:
        west_north_west(wind_direction, inflection_angle, t_dir, save, console)

    else:
        north_north_west(wind_direction, inflection_angle, t_dir, save, console)

# expos_damage uses output from Hurrecon and Expos to create a raster
# of hurricane wind damage where topograhic exposure at each location
# is determined by peak wind direction. If a location is protected, 
# the enhanced Fujita scale rating is reduced by two. This function 
# requires a hurricane tif file created by Hurrecon, eight exposure files 
# created by Expos (N, NE, E, etc), and a reprojection file in csv format 
# that contains lat long coordinates for the lower left and upper right 
# corners of the digital elevation model.
#   hurricane - hurricane name (as it appears in tif file)
#   inflection_angle - inflection angle (degrees)
#   save - whether to save results to file
#   console - whether to display messages in console
# returns a raster of landscape-level wind damage

def expos_damage (hurricane, inflection_angle, save=True, console=True):
    # get current working directory
    cwd = os.getcwd()

    # read exposure files as arrays
    ee_a = [None] * 8

    for i in range(0, 8):
        wind_direction = i*45

        expos_file = cwd + "/exposure/expos-" + str(wind_direction).zfill(3) + "-" + str(inflection_angle).zfill(2) + ".tif"

        ee_a[i] = rio.open(expos_file).read(1)

    # read dem file
    dem_file = cwd + "/dem/dem.tif"
    dem_r = rio.open(dem_file)

    dem_rows = dem_r.height
    dem_cols = dem_r.width

    # get dem array
    dem_a = dem_r.read(1)

    # get profile
    profile = dem_r.profile
    profile.update(dtype='int32', nodata=-9999, count=1)

    dem_r.close()

    # read hurrecon file
    hur_file = cwd + "/damage/" + hurricane + ".tif"
    hur_r = rio.open(hur_file)
    ff_a = hur_r.read(2)  # enhanced Fujita scale
    cc_a = hur_r.read(4)  # cardinal wind direction (1-8)

    hur_rows = hur_r.height
    hur_cols = hur_r.width

    hur_xmn = hur_r.bounds.left
    hur_xmx = hur_r.bounds.right
    hur_ymn = hur_r.bounds.bottom
    hur_ymx = hur_r.bounds.top

    hur_r.close()

    # read reproject file
    reproject_file = cwd + "/damage/reproject.csv"
    rp = pd.read_csv(reproject_file)

    lat_0 = rp.lat_0[0]
    lon_0 = rp.lon_0[0]
    lat_1 = rp.lat_1[0]
    lon_1 = rp.lon_1[0]

    # create damage array
    dam_a = np.where(dem_a != 0, 1, 0)

    # calculate exposure values
    for i in range(0, dem_rows):
        # display every 10th row number
        if i % 10 == 0:
            print("              ", end="")
            print("\rrow", i, end="")

        for j in range(0, dem_cols):
            # get lat long coordinates
            hur_x = lon_0 + (lon_1 - lon_0)*(j + 0.5)/dem_cols
            hur_y = lat_1 - (lat_1 - lat_0)*(i + 0.5)/dem_rows

            # get row & col in hurricane file (note: row 1 = top of raster)
            hur_row = math.ceil(hur_rows - hur_rows*(hur_y - hur_ymn)/(hur_ymx - hur_ymn))
            hur_col = math.ceil(hur_cols*(hur_x - hur_xmn)/(hur_xmx - hur_xmn))

            # skip missing values in dem
            if dem_a[i][j] != 0:
                # get peak wind direction (1-8)
                pdir = int(cc_a[hur_row][hur_col])

                if pdir == 0:
                    # no damage if no peak wind direction
                    dam_a[i][j] = 1

                else:
                    # get topographic exposure
                    exposure = int(ee_a[pdir-1][i][j])

                    # get fujita scale damage (0-7)
                    damage = int(ff_a[hur_row][hur_col])

                    # protected
                    if exposure == 1:
                        # reduce by two
                        pro_damage = damage - 2

                        if pro_damage < 1:
                            pro_damage = 1

                        dam_a[i, j] = pro_damage

                    # exposed
                    else:
                        dam_a[i][j] = damage

    if (save == True):
        # save modeled results in GeoTiff file
        dam_file = cwd + "/damage/" + hurricane + "-damage-" + str(inflection_angle).zfill(2) + ".tif"

        dam_tif = rio.open(dam_file, 'w', **profile)
        dam_tif.write(dam_a, 1)
        dam_tif.close()
   
        if console == True:
          print("\nSaving to", dam_file, "\n")

    else:
        # return modeled values as raster
        return dam_a


### SUMMARIZING FUNCTIONS #################################

# expos_summarize displays summary information for a specified raster
# file, including the number of rows and columns, spatial extent, cell
# height and width, and minimum and maximum value.
#   filename - name of input raster file
#   console - whether to display results in console
# returns a string containing summary information

def expos_summarize(filename, console=True):
    # get current working directory
    cwd = os.getcwd()
 
    # get subdirectory
    subdir = get_subdirectory(filename)

    # read file in GeoTiff format
    file_path = cwd + subdir + filename + ".tif"
    check_file_exists(file_path)
    rr = rio.open(file_path)

    # get number of rows & columns
    nrows = rr.height
    ncols = rr.width

    # get extent
    xmn = rr.bounds.left
    xmx = rr.bounds.right
    ymn = rr.bounds.bottom
    ymx = rr.bounds.top
    
    # get cell dimensions
    cell_x = (xmx-xmn)/ncols
    cell_y = (ymx-ymn)/nrows

    # get min & max values
    aa = rr.read()
    rr.close()
    
    val_min = aa.min()
    val_max = aa.max()

    # create display string
    st = "Rows: " + str(nrows) + "  Columns: " + str(ncols) + "\n"
    st = st + "Northing: " + str(round(ymn)) + " to " + str(round(ymx)) + "\n"
    st = st + "Easting: " + str(round(xmn)) + " to " + str(round(xmx)) + "\n"
    st = st + "Cell height: " + str(round(cell_y)) + "\n"
    st = st + "Cell width: " + str(round(cell_x)) + "\n"
    st = st + "Values: " + str(round(val_min, 1)) + " to " + str(round(val_max, 1)) + "\n"
    
    # display results in console
    if console == True:
        print(st)


### PLOTTING FUNCTIONS ####################################

# expos_plot creates a plot of a specified raster file. Optional arguments
# include plot title, horizontal units, vertical units, and color palette.
#   filename - name of input raster file
#   title - plot title
#   h_units - horizontal units
#   v_units - vertical units
#   vector - whether to display vectory boundary files
#   colormap - color palette
# no return value

def expos_plot(filename, title="", h_units="meters", v_units="meters",
    vector=True, colormap="default"):
    
    # get current working directory
    cwd = os.getcwd()
 
    # get subdirectory
    subdir = get_subdirectory(filename)

    # read file in GeoTiff format
    file_path = cwd + subdir + filename + ".tif"
    check_file_exists(file_path)
    rr = rio.open(file_path)

    rr_max = rr.read(1).max()

    # get vector boundary file
    if vector == True:
        boundaries_file = cwd + "/vector/boundaries.shp"
        shapefile = fiona.open(boundaries_file, "r")
        features = [feature["geometry"] for feature in shapefile]
        shapefile.close()

    # default palettes
    if colormap == "default":
        # exposure map
        if "expos" in filename:
            cols = ["grey", "blue"]
            cmap = matplotlib.colors.ListedColormap(cols)
            cmap.set_under('white')

        # damage map
        elif "damage" in filename:
            all_cols = ["grey", "purple", "blue", "green", "yellow", "orange", "red"]
            cols = []
            for i in range(0, rr_max):
                cols.append(all_cols[i])
            cmap = matplotlib.colors.ListedColormap(cols)
            cmap.set_under('white')

        # dem, etc
        else:
            cmap = plt.get_cmap("viridis")
            cmap.set_under('white')  
    
    # rasterio palette
    elif colormap == "r_default":
        cmap = plt.get_cmap("viridis")
        cmap.set_under('white')  

    # user-specified palette
    else:
        cmap = plt.get_cmap(colormap)
        cmap.set_under('white')

    # create plot
    if "dem" in filename:
        if title == "":
            title = "Elevation"
        v_units_str = "   " + v_units
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.xlabel(h_units)
        plt.ylabel(h_units)
        if vector == True:
            patches = [PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=1) for feature in features]
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True))
        img = rr.read(1)
        plt.title(title)
        plt.imshow(img, cmap=cmap, vmin=0.9)
        cbar = plt.colorbar(shrink=0.3)
        cbar.ax.set_title(v_units_str)
        show((rr, 1), cmap=cmap, vmin=0.9)
 
    elif "expos" in filename:
        if title == "":
            x = filename.split("-")
            title = "Exposure " + x[1] + "-" + x[2]
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.xlabel(h_units)
        plt.ylabel(h_units)
        if vector == True:
            patches = [PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=1) for feature in features]
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True))
        img = rr.read(1)
        plt.title(title)
        vals = [0, 1, 2]
        labs = ["", "Pro", "Exp"]
        plt.imshow(img, cmap=cmap, vmin=0.9)
        cbar = plt.colorbar(shrink=0.3)
        cbar.set_ticks(vals)
        cbar.set_ticklabels(labs)
        cbar.ax.set_title("   Expos")
        show((rr, 1), cmap=cmap, vmin=0.9)

    elif "damage" in filename:
        if title == "":
            x = filename.split("-")
            title = x[0] + " Damage " + x[2]
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.xlabel(h_units)
        plt.ylabel(h_units)
        if vector == True:
            patches = [PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=1) for feature in features]
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True))
        img = rr.read(1)
        plt.title(title)
        vals = [0, 1, 2, 3, 4, 5, 6, 7]
        labs = ["", "None", "EF0", "EF1", "EF2", "EF3", "EF4", "EF5"]
        plt.imshow(img, cmap=cmap, vmin=0.9)
        cbar = plt.colorbar(shrink=0.3)
        cbar.set_ticks(vals)
        cbar.set_ticklabels(labs)
        cbar.ax.set_title("   EF Scale")
        show((rr, 1), cmap=cmap, vmin=0.9)

    else:
        if title == "":
            title = filename
        fig, ax = plt.subplots(figsize=(15, 15))
        if vector == True:
            patches = [PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=1) for feature in features]
            ax.add_collection(mpl.collections.PatchCollection(patches, match_original=True))
        img = rr.read(1)
        plt.title(title)
        show((rr, 1), cmap=cmap, vmin=0.9)

    plt.clf()   
    rr.close()

