
"""

A module to write a met time series to a CSS point file 

The quickest way uses the entire array in memory (albeit briefly), 
with a loop version providing the slow mem-light alternative until I think
of something better

@author: ciaran robb
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
#from tqdm import tqdm
from osgeo import gdal#, ogr

gdal.UseExceptions()

# The ancillary functions

def _get_nc(inRas, lyr='rainfall'):
    """
    get the geotransform of netcdf
    
    for reference
    x_origin = rgt[0]
    y_origin = rgt[3]
    pixel_width = rgt[1]
    pixel_weight = rgt[5]

    """
    # As netcdfs are 'special(!)', need to specify the layer to get a rgt
    # rds = gdal.Open("NETCDF:{0}:{1}".format(inRas, lyr)) - this seems to apply
    # to MODIS but not the MET stuff
    
    rds = gdal.Open(inRas)

    
    rgt = rds.GetGeoTransform()
    
    img = rds.ReadAsArray()
    
    rds = None
    
    return rgt, img

def _points_to_pixel(gdf, rgt, espgin='epsg:27700', espgout='epsg:4326'):
    """
    convert some points from one proj to the other and return pixel coords
    and lon lats for some underlying raster
    
    returns both to give choice between numpy/pandas or xarray
    
    """
    
    xin = gdf.POINT_X
    yin = gdf.POINT_Y
    
    if espgin and espgout != None:
        transformer = Transformer.from_crs(espgin, espgout, always_xy=True) 
        # better than the old pyproj way
        points = list(zip(xin, yin))
        # output order is lon, lat
        coords_oot = np.array(list(transformer.itransform(points)))
    else:
        coords_oot = np.array([xin, yin])
        coords_oot = coords_oot.transpose()
    
    # for readability only
    lats = coords_oot[:,1]
    lons = coords_oot[:,0]
    
    
    # get the pixel coords using the rgt
    # v.minor risk of landing on 'wrong pixel' 
    px = np.int16((lons - rgt[0]) / rgt[1])
    py = np.int16((lats - rgt[3]) / rgt[5])
    
    return px, py, lons, lats


def _get_times(inRas):
    
    """
    get the datetimes stuff from the pd
    
    """
    
    # open it
    ds = xr.open_dataset(inRas)
    
    # header for the dataframe
    times = ds.time.to_dataframe()
    
    # what is xarray good for anyway
    del ds
    return times

    
    
# Main function

def met_time_series(inRas, inShp, outfile, prop, espgin=None, espgout=None):
    
    """ 
    Write met time series from a netcdf file to a point file
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    outfile: string
                  output shapefile
    
    prop: string
                 the propetry to be accessed e.g 'rainfall'
    
                     
    inmem: bool (optional)
                If True (default) the nc file will be loaded in mem as a numpy
                array to allow vectorised writing
    
    espgin: string
            a pyproj string for input proj of shapefile
            
    espgout: string
            a pyproj string for out proj of shapefile if reproj required
    """

    
    # Just in case I wish to move away from OGR shock horror....
    gdf = gpd.read_file(inShp)
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
        
    # the rgt and img array
    rgt, img = _get_nc(inRas, lyr=prop)
    
    # the rgt is enough with met data 
    px, py, lons, lats = _points_to_pixel(gdf, rgt, espgin=espgin, 
                                          espgout=espgout)
    
    # for ref either a shapely geom or the entry in pd
    #mx, my = gdf.geometry[0].coords[0] or gdf.POINT_X, gdf.POINT_Y
    
    times = _get_times(inRas)
    
    # cut the year so we can give a name
    cols = list(times.index.strftime("%y-%m"))
    
    cols = [prop[0:4]+"-"+c for c in cols]
    
    # There are 2 options - vectorised or python loop 
    # if we don't care about memory footprints we use xarray to get at the
    # np structure - fine if the data is not enormous
    # could be costly if it is many years, but met images are 
    # tiny
    
    # quickest w/gdal/np inds bnds,y,x    
    ndvals = img[:, py, px]
    ndvals = np.swapaxes(ndvals, 0, 1)
    
    nddf = pd.DataFrame(data=ndvals, index=gdf.index,
                        columns=cols)

    # why is there a duplicate value key_0 for index ?
    #- returns a bug of  duplicate values
    # index without this long param list to hack around it
    newdf = pd.merge(gdf, nddf, how='left', on=gdf.index)
        
    
    newdf.to_file(outfile)



def tseries_group(df, name, other_inds=None):
    
    """
    Extract time series of a particular variable e.g. rain
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    name: string
          the identifiable string e.g. rain, which can be part or all of 
          column name
                      
    year: string
            the year of interest
    
    other_inds: string
            other columns to be included 

    """
    # probably a more elegant way...but works
    ncols = [y for y in df.columns if name in y]
    
    # if we wish to include something else
    if other_inds != None:
        ncols = other_inds + ncols
        
    
    newdf = df[ncols]
    
    return newdf


def plot_group(df, group, index, name):
    
    """
    Plot time series per CSS square eg for S2 ndvi or met var
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: int
            the index of interest
            
    name: string
            the name of interest

    
    """
    
    # Quick dirty time series plotting

    sqr = df[df[group]==index]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    ndplotvals = sqr[yrcols]
    
    ndplotvals.transpose().plot.line()
    
    #return ndplotvals


























