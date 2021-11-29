## EXPOS Model

The EXPOS model estimates topographic exposure to hurricane winds as a function
of wind direction and inflection angle.

The input file is assumed to be a raster file in GeoTiff format with missing values
represented by zero.  Cells may be rectangular but horizontal and vertical units 
must be the same. Columns are assumed to be closely aligned with true north.
The name of the input file is assumed to be "dem.tif".

The output file is a raster file in GeoTiff format with the following values: 
0 = missing data, 1 = protected, 2 = exposed. Output files are named "expos-xxx-yy.tif"
where xxx is the wind direction and yy is the inflection angle.

In previous studies, a spatial resolution of 30 meters and an inflection angle of 
6 degrees were found to work well (see below). Note that increasing the inflection 
angle tends to decrease the size and number of protected areas while increasing 
the odds that these sites are protected.

EXPOS is available in both R (ExposR) and Python (ExposPython) versions. 
The model is an updated version of the original EXPOS model written in Borland 
Pascal for use with Idrisi (see below for details).

Please note: both versions are currently under development and subject to change.

## Getting Started

Here are the basic steps for using the model. Please see below for more details.

1. Download the R or Python version of the model from GitHub.
2. Create a directory for a particular study region.
3. Copy the digital elevation file to this directory and rename it "dem.tif".
4. Run the model to create exposure maps for different wind directions and inflection
angles.

## Details

All user functions begin with "expos". The wind direction and inflection angle must be
specified in degrees. Input and output files for a given region are located on the same
directory.

To run the model, run expos.R (R) or expos.py (Python). 

The R version may also be installed as an R package using the devtools package:

```{r}	
devtools::install_github("expos-model/ExposR")
```

This has the advantage of providing readily accessible help messages for each 
function.

## Model Functions

```{r}	
expos_set_path
expos_model
expos_damage
expos_summarize
expos_plot
```

The expos_set_path function sets the path for the current set of model runs.

The expos_model function creates a raster file of wind exposure as a function
of wind direction and inflection angle.

The expos_damage function uses output from Hurrecon and Expos to create a raster
file of wind damage where topograhic exposure at each location is determined 
by peak wind direction. If a location is protected, the enhanced Fujita scale 
rating is reduced by two.

The expos_summarize function displays summary information for a specified raster
file, including the number of rows and columns, spatial extent, cell height and 
width, and minimum and maximum value.

The expos_plot function creates a plot of a specified raster file.

## Examples


```{r}
expos_set_path("c:/expos/r/mass_30m")
expos_model(wind_direction=90, inflection_angle=6)
expos_damage("AL061938", inflection_angle=6)
expos_summarize("dem")
expos_plot("expos-090-06")
```

## History

The original EXPOS model was written in Borland Pascal and depended on Idrisi 
for spatial visualization. The model was used in published studies of the ecological 
impacts of historical hurricanes in New England and Puerto Rico:

* Boose, E. R., Chamberlin, K. E., Foster, D. R. 2001. Landscape and regional impacts 
of hurricanes in New England. Ecological Monographs 71: 27-48.

* Boose, E. R., Serrano, M. I., Foster, D. R. 2004. Landscape and regional impacts of 
hurricanes in Puerto Rico. Ecological Monographs 74: 335-352.

