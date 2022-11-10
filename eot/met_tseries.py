
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

def _get_nc(inRas, lyr=None, im=True):
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
    
    if im == True:
        img = rds.ReadAsArray()
        rds = None
        return rgt, img
    
    else:
        rds = None
        return rgt
        
def _points_to_pixel(gdf, rgt, espgout='epsg:4326'):
    """
    convert some points from one proj to the other and return pixel coords
    and lon lats for some underlying raster
    
    returns both to give choice between numpy/pandas or xarray
    
    """
    
    xin = gdf.POINT_X
    yin = gdf.POINT_Y
    
    if espgout != None:
        gdf = gdf.to_crs({'init': espgout})
        
    
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

def met_time_series(inRas, inShp, outfile, prop, espgout=None):
    
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
    px, py, lons, lats = _points_to_pixel(gdf, rgt, 
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
    # this isn't quite right....
    ndvals = img[:, py, px]
    ndvals = np.swapaxes(ndvals, 0, 1)
    
    nddf = pd.DataFrame(data=ndvals, index=gdf.index,
                        columns=cols)
    

    # why is there a duplicate value key_0 for index ?
    #- returns a bug of  duplicate values
    # index without this long param list to hack around it
    #newdf = pd.merge(gdf, nddf, how='left', on=gdf.index)
    
    newdf = pd.concat([gdf, nddf], axis=1)
        
    
    newdf.to_file(outfile)

def met_time_series_to_sheet(inRas, inShp,  prop, espgout=None):
    
    """ 
    Get met time series from a point file and write to an xls sheet, the object
    to have a seperate sheet per met variable
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRasList: string
                  input raster

    outfile: string
                  output shapefile
    
    prop: string
                 the propetry to be accessed e.g 'rainfall'
    
    
    espgin: string
            a pyproj string for input proj of shapefile
            
    espgout: string
            a pyproj string for out proj of shapefile if reproj required
    """

    
    # Just in case I wish to move away from OGR shock horror....
    
    gdf = gpd.read_file(inShp)
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
        
    # the rgt
    # must remember to index the layer the gdal way,
    # so the first band would be [0,:,:]
    rgt, img = _get_nc(inRas, lyr=prop)
    # this will yield more intuitive dims but makes no dif 
    #img = img.transpose((1,2,0))
    
    # the rgt is enough with met data 
    px, py, lons, lats = _points_to_pixel(gdf, rgt,  
                                          espgout=espgout)
    
    # for ref either a shapely geom or the entry in pd
    #mx, my = gdf.geometry[0].coords[0] or gdf.POINT_X, gdf.POINT_Y
    
    times = _get_times(inRas)
    
    # cut the year so we can give a name
    cols = list(times.index.strftime("%y-%m"))
    
    # quickest w/gdal/np inds [bnds,y,x]

    ndvals = img[:, py, px]
    ndvals = np.swapaxes(ndvals, 0, 1)
    
    nddf = pd.DataFrame(data=ndvals, index=gdf.index,
                        columns=cols)
    
    newdf = pd.concat([gdf.REP_ID, nddf], axis=1)
    
    return newdf


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

def rasterize(inShp, inRas, outRas, field=None, fmt="Gtiff"):
    
    """ 
    Rasterize a polygon to the extent & geo transform of another raster


    Parameters
    -----------   
      
    inRas: string
            the input image 
        
    outRas: string
              the output polygon file path 
        
    field: string (optional)
             the name of the field containing burned values, if none will be 1s
    
    fmt: the gdal image format
    
    """
    
    
    
    inDataset = gdal.Open(inRas)
    
    # the usual 
    
    outDataset = _copy_dataset_config(inDataset, FMT=fmt, outMap=outRas,
                         dtype = gdal.GDT_Int32, bands=1)
    
    
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    
    if field == None:
        gdal.RasterizeLayer(outDataset, [1], lyr, burn_values=[1])
    else:
        gdal.RasterizeLayer(outDataset, [1], lyr, options=["ATTRIBUTE="+field])
    
    outDataset.FlushCache()
    
    outDataset = None


def _copy_dataset_config(inDataset, FMT = 'Gtiff', outMap = 'copy',
                         dtype = gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
    x_pixels = inDataset.RasterXSize  
    y_pixels = inDataset.RasterYSize  
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  

    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    #dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    outDataset = driver.Create(
        outMap, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    outDataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))
        
    outDataset.SetProjection(projection)
    
    return outDataset


def _create_raster(xsize, ysize, driver='MEM', tmpfile='', gt=None, srs_wkt=None,
                  nodata=None,  init=None, datatype=gdal.GDT_Byte):
    """
    Create a raster 
    """
    
    out_ds = gdal.GetDriverByName(driver).Create(tmpfile, xsize, ysize, 1, datatype)
    
    # should it be necessary to init the raster with some value
    if init is not None:
        out_ds.GetRasterBand(1).Fill(init)
        
    if nodata is not None:
        out_ds.GetRasterBand(1).SetNoDataValue(nodata)
        
    if gt:
        out_ds.SetGeoTransform(gt)
        
    if srs_wkt:
        out_ds.SetProjection(srs_wkt)
        
    return out_ds


def rasterize_point(inshp, outras, exp, field, pixel_size=5000, dtype=gdal.GDT_Float32):
    
    """ 
    Rasterize/grid a point file specifying the attribute to be gridded
    Esoteric - written to process not clever data from CEH

    Parameters
    -----------   
      
    inshp: string
            the input point file
        
    outras: string
              the output polygon file path 
        
    exp: string 
             the expression "year=1986"
    
    field: string
            the field values to be raterised eg 'NOx'
    
    pixel_size: int
                assumes a square pixel (IMPORTANT - gridding from top left)
    
    dtype: int
            a gdal dtype e.g. gdal.GDT_Float32 which = 6
    
    """
    
    vds = ogr.Open(inshp)
    
    lyr = vds.GetLayer()
    
    xmin, xmax, ymin, ymax = lyr.GetExtent()
    
    ext = xmin, ymin, xmax, ymax
    
    # require an input dataset but must work out from res and ext
    tr = pixel_size # space saving 
    
    out_gt = (ext[0], tr, 0, ext[3], 0, -tr)
    
    cols = int(math.ceil((ext[2] - ext[0]) / tr))
    rows = int(math.ceil((ext[3] - ext[1]) / tr))
    
    # this not required if doing via the comm'd out alternative
    lyr.SetAttributeFilter(exp)
    
    ref = lyr.GetSpatialRef()
    
    rds = _create_raster(cols, rows, driver='Gtiff', tmpfile=outras, 
                           gt=out_gt, srs_wkt=ref.ExportToWkt(), nodata=None, 
                           datatype=dtype)
    
    gdal.RasterizeLayer(rds, [1], lyr, options=["ATTRIBUTE="+field])
    
    # This is another way of doing the same, albeit the selection
    # is less simply written and less python api use - good to ken for ref
    # this works as above, with the same issue on larger datasets
    #ops = gdal.RasterizeOptions(attribute=field, where='"year"=\'1986\'')
    #gdal.Rasterize(rds, inshp, options=ops)
    
    rds.FlushCache()
    rds = None
    vds = None

















