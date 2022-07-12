## EXPOS Model

The EXPOS model uses a digital elevation model (DEM) to estimate exposed and
protected areas for a given hurricane wind direction and inflection angle. 
The resulting topograhic exposure maps can be combined with output from the 
HURRECON model to estimate hurricane wind damage across a region.

EXPOS contains two main functions:

1. The <i>expos_model</i> function estimates topographic exposure for a specified
wind direction and inflection angle. The input file is assumed to be a raster of 
elevation values in GeoTiff format with missing values represented by zero. Cells 
may be rectangular. The user can specify if coordinates are lat/long; otherwise 
lat/long is assumed if X values are between -180 and 180 and Y values are between
-90 and 90. If coordinates are lat/long, horizontal and vertical units are assumed 
to be degrees and meters, respectively; otherwise horizontal and vertical units 
must be the same. Columns are assumed to be closely aligned with true North (0 
degrees); if not, the map orientation (azimuth) must be specified in degrees. 
The name of the input file is assumed to be "dem.tif".

The output file is a raster file in GeoTiff format with the following values: 
0 = missing data, 1 = protected, 2 = exposed. Output files are named 
"expos-xxx-yy.tif" where xxx is the wind direction and yy is the inflection angle.

In previous studies, spatial resolutions of 30 or 60 meters and an inflection angle of 
6 degrees were found to work well (see below). Note that increasing the inflection 
angle tends to decrease the size and number of protected areas while increasing 
the odds that these sites are protected.

2. The <i>expos_damage</i> function estimates regional hurricane damage where
topographic exposure at each location is determined by peak wind direction. If 
a location is protected, the enhanced Fujita scale (EF) rating from HURRECON 
is reduced by a specified number of EF ratings. This function requires a hurricane 
GeoTiff file created by HURRECON, exposure files created by EXPOS for the eight 
cardinal wind directions (N, NE, E, etc), and a reprojection file in CSV format 
(<i>reproject.csv</i>) that contains lat/long coordinates for the lower left and 
upper right corners of the digital elevation model (variables: <i>name</i>, 
<i>lat_0</i>, <i>lon_0</i>, <i>lat_1</i>, <i>lon_1</i>).

The output file is a raster file in GeoTiff format with the following values:
0 = missing, 1 = no damage, 2 = EF0 damage, 3 = EF1 damage, 4 = EF2 damage, 
5 = EF3 damage, 6 = EF4 damage, 7 = EF5 damage. Output files are named 
"hhhh-damage-yy-z.tif" where hhhh is the hurricane ID, yy is the inflection 
angle, and z is the reduction in EF rating for protected areas.

EXPOS is available in both R (ExposR) and Python (ExposPython) versions. 
The model is an updated version of the original EXPOS model written in Borland 
Pascal for use with Idrisi (see below for details).

Please note: both versions are currently under development and subject to change.

## Getting Started

Here are the basic steps for using the model. Please see below for more details.

1. Download the R or Python version of the model from GitHub.
2. Create a directory for a particular study area with subdirectories as described 
below.
3. Copy the digital elevation file to the <i>dem</i> subdirectory and rename it 
"dem.tif".
4. Download or create geographic and political boundary shapefiles for the desired 
study area. Copy these files to the <i>vector</i> subdirectory and rename so the 
first name of each file is "boundaries".
5. Use the <i>expos_model</i> function to create exposure maps for different wind 
directions and inflection angles.
6. Use the <i>expos_damage</i> function to create maps of enhanced Fujita scale 
wind damage for particular hurricanes.

## Details

All user functions begin with "expos". The wind direction and inflection angle must
be specified in degrees.

The user specifies a directory (<i>exp_path</i>) for a given study area. Input and 
output files are stored on the following subdirectories of this directory:

```
exp_path/dem
exp_path/exposure
exp_path/damage
exp_path/vector
```

The <i>dem</i> subdirectory contains the input elevation file. The <i>exposure</i> 
subdirectory contains output files from the <i>expos_model</i> function. The 
<i>damage</i> subdirectory contains input files from the HURRECON model, the 
reprojection file, and output files from the <i>expos_damage</i> function. 
Shapefiles that contain geographic and political boundaries for viewing results
are stored on the <i>vector</i> subdirectory.

To run the model, run <i>expos.R</i> (R) or <i>expos.py</i> (Python). 

The R version may also be installed as an R package using the devtools package:

```
devtools::install_github("expos-model/ExposR")
```

This has the advantage of providing readily accessible help messages for each 
function.

## Model Functions

```
expos_set_path
expos_get_path
expos_model
expos_damage
expos_summarize
expos_plot
```

The <i>expos_set_path</i> function sets the path for the current set of model runs. 
The <i>expos_get_path</i> function returns the current path. Use <i>expos_set_path</i> 
before using other functions.

The <i>expos_model</i> function creates a raster file of topographic wind exposure 
as a function of wind direction and inflection angle.

The <i>expos_damage</i> function uses output from EXPOS and HURRECON to create a 
raster file of wind damage where topograhic exposure at each location is determined 
by peak wind direction. If a location is protected, the enhanced Fujita scale 
rating is reduced by a specfied amount.

The <i>expos_summarize</i> function displays summary information for a specified 
raster file, including the number of rows and columns, spatial extent, cell height 
and width, and minimum and maximum value.

The <i>expos_plot</i> function creates a plot of a specified raster file.

## Examples

```
expos_set_path("c:/expos/wach_30m")
expos_get_path()

expos_model(wind_direction=90, inflection_angle=6)
expos_damage(hurricane="AL1938-06", inflection_angle=6, protect=2)

expos_summarize("dem")

expos_plot("dem")
expos_plot("expos-090-06")
expos_plot("AL1938-06-damage-06-2")
```

## History

The original EXPOS model was written in Borland Pascal and depended on Idrisi 
for spatial visualization. The model was used in published studies of the ecological 
impacts of historical hurricanes in New England and Puerto Rico:

* Boose, E. R., Foster, D. R., Fluet, M. 1994. Hurricane impacts to tropical and 
temperate forest landscapes. Ecological Monographs 64: 369-400. doi:10.2307/2937142.

* Boose, E. R., Chamberlin, K. E., Foster, D. R. 2001. Landscape and regional impacts 
of hurricanes in New England. Ecological Monographs 71: 27-48.
doi:10.1890/0012-9615(2001)071[0027:LARIOH]2.0.CO;2.

* Boose, E. R., Serrano, M. I., Foster, D. R. 2004. Landscape and regional impacts of 
hurricanes in Puerto Rico. Ecological Monographs 74: 335-352. doi:10.1890/02-4057.

