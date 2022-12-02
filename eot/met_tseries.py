
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
from osgeo import gdal, ogr
import math
from tqdm import tqdm
import geopandas as gpd
import netCDF4 as nc
ogr.UseExceptions()
gdal.UseExceptions()

# The ancillary functions

def create_nc(array, rgt, start=1986, end=2012):
    
    per = end-start+1
    drange = pd.date_range(start=pd.datetime(start, 1, 1), periods=per,
                           freq='A')
    #I need to do an arange to get the coords (assumming that xarray
    # operates like gdal)
    
    
    
    crd = dict(time=drange)
    ds = xr.DataArray(data=array,  dims=["x", "y", "time"],
                      coords=crd)
    
def _raster_extent(inras):
    
    """
    Parameters
    ----------
    
    inras: string
        input gdal raster (already opened)
    
    """
    rds = gdal.Open(inras)
    rgt = rds.GetGeoTransform()
    minx = rgt[0]
    maxy = rgt[3]
    maxx = minx + rgt[1] * rds.RasterXSize
    miny = maxy + rgt[5] * rds.RasterYSize
    ext = (minx, miny, maxx, maxy)
    
    return ext

def _get_rgt(inras):
    
    """
    get rgt
    """
    
    rds = gdal.Open(inras)

    
    rgt = rds.GetGeoTransform()
    
    return rgt
    
    

def _get_nc(inRas, lyr=None, im=True):
    """
    get the geotransform of netcdf
    
    for reference
    x_origin = rgt[0]
    y_origin = rgt[3]
    pixel_width = rgt[1]
    pixel_weight = rgt[5]

    """
    # As netcdfs are 'special, need to specify the layer to get a rgt
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
    
    inShp: string131
                  input shapefile
        
    inRas: string
                  input raster
    
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
    #rgt, img = _get_nc(inRas, lyr=prop)
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


def met_time_series_to_sheet2(inShp, array, rgt, prop, times, espgout=None):
    
    """ 
    Get met time series from a point file and write to an xls sheet, the object
    to have a seperate sheet per met variable
    
    Parameters
    ----------
    
    inShp: string131
                  input shapefile
        
    array: np array
                  3d numpy arrau

    rgt: list
                a gdal raster geotransform
    
    prop: string
                 the propetry to be accessed e.g 'rainfall'
            
    espgout: string
            a pyproj string for out proj of shapefile if reproj required
    """

    
    # Just in case I wish to move away from OGR shock horror....
    
    # think this should be outside func as too slow and no good for para
    if isinstance(inShp, str) == True:
        gdf = gpd.read_file(inShp)
    else:
        # could add this...
        # isinstance(inshp, geopandas.geodataframe.GeoDataFrame)
        gdf = inShp

    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
        
    # the rgt
    # must remember to index the layer the gdal way,
    # so the first band would be [0,:,:]
    
    #rgt, img = _get_nc(inRas, lyr=prop)
    # this will yield more intuitive dims but makes no dif 
    #img = img.transpose((1,2,0))
    
    # the rgt is enough with met data 
    px, py, lons, lats = _points_to_pixel(gdf, rgt,  
                                          espgout=espgout)
    
    # for ref either a shapely geom or the entry in pd
    #mx, my = gdf.geometry[0].coords[0] or gdf.POINT_X, gdf.POINT_Y
    
    #times = _get_times(inRas)
    
    # cut the year so we can give a name
    #cols = list(times.index.strftime("%y-%m"))
    
    # quickest w/gdal/np inds [bnds,y,x]
    # the other way with the chem data whoops
    ndvals = array[py, px, :]
    #ndvals = np.swapaxes(ndvals, 0, 1)
    
    nddf = pd.DataFrame(data=ndvals, index=gdf.index,
                        columns=times)
    
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
    """
    Copies a dataset without the associated rasters.

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
    Create a raster with the usual options
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


def rasterize_point(inshp, exp, field, outras=None,
                     pixel_size=5000, dtype=gdal.GDT_Float32):
    
    """ 
    Rasterize/grid a point file specifying the attribute to be gridded
    Esoteric? - written to process not clever data from UKCEH

    Parameters
    -----------   
      
    inshp: string
            the input point file
        
    exp: string 
             the selection expression "year=1986"
             
    outras: string
              the output raster file path, if None, an array will be returned
    
    field: string
            the field values to be raterised eg 'NOx'
    
    pixel_size: int
                assumes a square pixel (IMPORTANT - gridding from top left)
    
    dtype: int
            a gdal dtype e.g. gdal.GDT_Float32 which = 6
    
    Returns
    
    array, raster geo-transform
    
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
    
    if outras is None:
        drv = 'MEM'
        ootras = "" # if this was outras it'd ruin the later if statement!!!
    else:
        drv = 'Gtiff'
    
    rds = _create_raster(cols, rows, driver=drv, tmpfile=ootras, 
                           gt=out_gt, srs_wkt=ref.ExportToWkt(), nodata=None, 
                           datatype=dtype)
    
    gdal.RasterizeLayer(rds, [1], lyr, options=["ATTRIBUTE="+field])
    
    # This is another way of doing the same, albeit the selection
    # is less simply written and less python api use and avoids opening the shp
    # file - good to ken for ref 
    # the use of different quotes above is key to distinguish a field ("year")
    # from a value (\'1986\'') - add to this python string conventions
    # comapred top cmd line and it all becomes a bit confusing
    #ops = gdal.RasterizeOptions(attribute=field, where='"year"=\'1986\'')
    #gdal.Rasterize(rds, inshp, options=ops)
    
    if outras is None:
        img = rds.GetRasterBand(1).ReadAsArray()
        rgt = rds.GetGeoTransform()
        return img, rgt
    else:
        rds.FlushCache()
        
    rds = None
    vds = None

def create_chem_stk(inshp, field,  pixel_size=5000, dtype=6):
    
    """
    Create a 3d array of chemical deposition, where years are the 3rd/Z 
    dimension
    
    Parameters
    -----------   
      
    inshp: string or gpd
            the input point file
        
    field: string
            the field values to be raterised eg 'NOx'
    
    pixel_size: int
                assumes a square pixel (IMPORTANT - gridding from top left)
    
    Returns
    -------
    years, nparray
    
    """
    # think this should be outside func as too slow and no good for para
    # if isinstance(inshp, str) == True:
    gdf = gpd.read_file(inshp)
    # else:
    #     # could add this...
    #     # isinstance(inshp, geopandas.geodataframe.GeoDataFrame)
    #gdf = inshp
    
    yrs  = list(gdf.year.unique()) 
    
    # get the dims of the img derived from gridded points
    exp1 = "year="+ str(yrs[0])
    tmp, _ = rasterize_point(inshp, exp1, field, outras=None, 
                             pixel_size=5000, dtype=6)
    
    rsshape = (tmp.shape[0], tmp.shape[1], len(yrs))
    farray = np.zeros(shape=rsshape)
    

    for idx, y in tqdm(enumerate(yrs)):
        exp = "year="+ str(y)
        img, _ = rasterize_point(inshp, exp, field,
                                 outras=None, pixel_size=5000,
                                 dtype=6)
        farray[:, :, idx] = img
    
    return yrs, farray


def _make_gdflist(flist):
    
    """
    make a list of gdfs from stupid ceh data formats
    """
    
    gdfs = []
    
    for f in tqdm(flist):
        df = pd.read_csv(f)
        gd = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.easting, df.northing))
        gdfs.append(gd)
        
    return gdfs



def array2raster(array, bands, inRaster, outRas, dtype, FMT=None):
    
    """
    Save a raster from a numpy array using the geoinfo from another.
    
    Parameters
    ----------      
    array: np array
            a numpy array.
    
    bands: int
            the no of bands. 
    
    inRaster: string
               the path of a raster.
    
    outRas: string
             the path of the output raster.
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA.
        
    
    """

    if FMT == None:
        FMT = 'Gtiff'
        
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'    
    
    inras = gdal.Open(inRaster, gdal.GA_ReadOnly)    
    
    x_pixels = inras.RasterXSize  # number of pixels in x
    y_pixels = inras.RasterYSize  # number of pixels in y
    geotransform = inras.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    #if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inras.GetProjection()
    geotransform = inras.GetGeoTransform()   

    driver = gdal.GetDriverByName(FMT)

    dataset = driver.Create(
        outRas, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))    

    dataset.SetProjection(projection)
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
    else:
    # Here we loop through bands
        for band in range(1,bands+1):
            Arr = array[:,:,band-1]
            dataset.GetRasterBand(band).WriteArray(Arr)
        dataset.FlushCache()  # Write to disk.
        dataset=None
        #print('Raster written to disk')
        
def raster2array(inRas, bands=[1]):
    
    """
    Read a raster and return an array, either single or multiband

    
    Parameters
    ----------
    
    inRas: string
                  input  raster 
                  
    bands: list
                  a list of bands to return in the array
    
    """
    rds = gdal.Open(inRas)
   
   
    if len(bands) ==1:
        # then we needn't bother with all the crap below
        inArray = rds.GetRasterBand(bands[0]).ReadAsArray()
        
    else:
        #   The nump and gdal dtype (ints)
        #   {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,"uint32": 4,"int32": 5,
        #    "float32": 6, "float64": 7, "complex64": 10, "complex128": 11}
        
        # a numpy gdal conversion dict - this seems a bit long-winded
        dtypes = {"1": np.uint8, "2": np.uint16,
              "3": np.int16, "4": np.uint32,"5": np.int32,
              "6": np.float32,"7": np.float64,"10": np.complex64,
              "11": np.complex128}
        rdsDtype = rds.GetRasterBand(1).DataType
        inDt = dtypes[str(rdsDtype)]
        
        inArray = np.zeros((rds.RasterYSize, rds.RasterXSize, len(bands)), dtype=inDt) 
        for idx, band in enumerate(bands):  
            rA = rds.GetRasterBand(band).ReadAsArray()
            inArray[:, :, idx]=rA
   
   
    return inArray









