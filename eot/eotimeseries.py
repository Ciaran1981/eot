"""
A few functions for extracting of time series from S1/S2 data with GEE with a

view to comparing it with something else 

@author: Ciaran Robb

"""
import geemap
import pandas as pd
from datetime import datetime
from pyproj import Transformer
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from osgeo import ogr, osr, gdal
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import chain
import ee, eemont, wxee
from eot.s2_masks import addCloudShadowMask, applyCloudShadowMask, addGEOS3Mask, loadImageCollection
from eot.s2_fcover import fcover
from eot.composites import*
import math
from shapely.geometry import box, mapping
ee.Initialize()

# stop this warning - concerns bs_ funcs at present
# A value is trying to be set on a copy of a slice from a DataFrame
# See the caveats in the documentation:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.chained_assignment = None
ogr.UseExceptions()
osr.UseExceptions()

def get_collection_dates(collection):
    
    """
    Extract the dates from an image collection in a palatable format
    
    Parameters
    ----------
    
    collection: 
                 a pre-constructed gee 
    
    """
    
    def dfunc(image):
        return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})

    dates = collection.map(dfunc).distinct('date').aggregate_array('date')
    return dates


def clip_collection(collection, geom):
    
    """
    Clip a collection with a geometry
    
    Parameters
    ----------
    
    collection: 
                 a pre-constructed gee collection
    
    geom:
          an gee geometry
    
    """
    
    def _clip(image):
        return image.clip(geom)
    return collection.map(_clip)

# TODO, this could just be added to s2cloudless
def geos3(collection):

    """
    Add a geos3 bare soil band to a S2 collection
    
    Parameters
    ----------
    
    collection: 
                 a pre-constructed gee collection
              
    start_date: string
                    start date of time series
                    
    end_date: string
                    end date of time series
    
    geom: string
          an ogr compatible file with an extent
          
          
    Returns
    -------
    
    S2 collection with geos3 mask band (binary) included
    
    """
    
    def _funcbs(img):
        
        geos3 = addGEOS3Mask(img)
        return img.addBands(geos3)
        # originally this updated the mask meaning only the bare pixels were returned for all bands
        # return img.updateMask(geos3)
        return img.addBands(geos3)

    bs_collection = collection.map(_funcbs)
    return bs_collection

def _funcbsnotmask(img):
    geos3 = addGEOS3Mask(img)
    return img.updateMask(geos3)

def baresoil_collection(inshp, start_date='2021-01-01', end_date='2021-12-31',
                        band_list=['fcover']):
    """
    Generate bare soil(bs) layers using google earth engine with the GEOS3
    method
    
    Parameters
    ----------
    
    inshp: string or json type dict
        Path to input vector file or geojson/dict geometry
        
    start_date: string
                    start date of time series
                    
    end_date: string
                    end date of time series  
                    
    year: string
        The year to return a collection from. The default is '2021'.

    Returns
    -------
    A tuple containing bs_masked, bs_collection, bs_freq

    """
    
    
    
    if isinstance(inshp, dict):
        # if it is just a geojson geom carry on
        geom = inshp 
    # this need for zonal ts as we pass a bbox ee.geometry
    elif isinstance(inshp, ee.geometry.Geometry):
        geom = inshp
        
    else:
        # we assume (probably shouldn't) it is a string
        # get the shp file ext to demarcate the collection
        county = extent2poly(inshp, filetype='polygon', outfile=False, 
                         polytype="ESRI Shapefile",  geecoord=True)
        geom = county
    
    years = ee.Dictionary({'2016': 'COPERNICUS/S2',
                               '2017': 'COPERNICUS/S2',
                               '2018': 'COPERNICUS/S2',
                               '2019': 'COPERNICUS/S2_SR',
                               '2020': 'COPERNICUS/S2_SR',
                               '2021': 'COPERNICUS/S2_SR'})

    # date range dict
    dts = {'start': start_date, 'end': end_date}
    date_range_temp = ee.Dictionary(dts)
    year = start_date[0:4]

    # Load the Sentinel-2 collection for the time period and area requested
    s2_cl = loadImageCollection(ee.String(years.get(year)).getInfo(), 
                                date_range_temp, geom)
 
    masked_collection = s2cloudless(s2_cl, dts['start'], dts['end'], geom,
                    cloud_filter=60)
    
    # handy for now (cheers soilwatch - credit to them), but likely to be replaced
    # by uk datasets
    # Import external mask datasets
    # JRC Global Surface Water mask
    not_water = ee.Image("JRC/GSW1_2/GlobalSurfaceWater").select('max_extent').eq(0) 
    # JRC global human settlement layer
    jrc_builtup = ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1")
    # Facebook Population Layer under MIT License, courtesy of: Copyright (c) 2021 Samapriya Roy
    facebook_builtup = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
    # Combine JRC builtup and facebook population as builtup mask
    not_builtup = jrc_builtup.select('built').gt(2) \
                      .bitwiseOr(facebook_builtup.mosaic().unmask(0).gt(1)) \
                      .Not()
    
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    dem_proj = dem.first().projection()
    dem = dem.mosaic().setDefaultProjection(dem_proj)

    slope_deg = ee.Terrain.slope(dem)
    #slope_rad = slope_deg.multiply(ee.Image(math.pi).divide(180))
    #slope_aspect = ee.Terrain.aspect(dem)
    #LS = factorLS(slope_deg, slope_rad, slope_aspect)
    
    # TODO does this need to be 2 collections?
    # for the freq calculation
    bs_collection = masked_collection.map(_funcbsnotmask)
    # ...and the one with a binary mask wherever there is bare soil.
    bs_masked = geos3(masked_collection)
    
    # Set base date to generate a Day of Year layer
    
    #from_date = ee.Date.parse('YYYY-MM-dd', year + '-01-01')

    # # Generate the series to be used as input for the drawing tool plot.
    # plot_series = drawingTools.preparePlotSeries(masked_collection, bs_collection, geom,
    #                                                  from_date, date_range_temp, band_list)

    date_range = date_range_temp

    # Apply the actual date range specified, as opposed to the full year (required for the drawing tools plotting)
    masked_collection = masked_collection.filterDate(date_range.get('start'), date_range.get('end'))

    bs_collection = bs_collection.filterDate(date_range.get('start'), date_range.get('end'))
    #v_collection = v_collection.filterDate(date_range.get('start'), date_range.get('end'))

    # Generate a list of time intervals for which to generate a harmonized time series
    time_intervals = extractTimeRanges(date_range.get('start'), date_range.get('end'), 30)

    #TODO - post translation the harmonic series is not working - 
    # not critical but it'd be nice to ken 
    # the naming problem occurs here!! All fine until this point
    # Generate harmonized monthly time series of FCover as input to the vegetation factor V
    #fcover_ts = harmonizedTS(masked_collection, band_list, time_intervals, band_name='fcover')#, {'agg_type': 'geomedian'})

    # Run a harmonic regression on the time series to fill missing data gaps and smoothen the NDVI profile.
    #fcover_ts_smooth = harmonicRegression(fcover_ts, 'fcover', 4)
                                     # clamping to [0,10000] data range,
                                     # as harmonic regression may shoot out of data range

    # Calculate the bare soil frequency,
    # i.e. the number of bare soil observations divided by the number of cloud-free observations
    bs_freq = bs_collection.select('B2').count() \
                  .divide(masked_collection.select('B2').count()) \
                  .rename('bare_soil_frequency') \
                  .clip(geom) \
                  .updateMask(not_water.And(not_builtup).And(slope_deg.lte(26.6)))
                  
    # TODO Update! fine for now but perhaps improvable
    # Define a mask that categorizes pixels > 95% frequency as permanently bare,
    # i.e. rocky outcrops and other bare surfaces that have little to no restoration potential
    bs_freq_mask = bs_freq.gt(0).And(bs_freq.lt(0.95))

    #area_chart =  charts.areaChart(bs_freq, 'bare_soil_frequency', geom, pie_options)

    bs_freq = bs_freq.updateMask(bs_freq_mask)
    
    return bs_masked, bs_collection, bs_freq
  



def s2cloudless(collection, start_date, end_date, geom,
                cloud_filter=60):
    
    """
    Make a S2 cloudless collection
    
    Parameters
    ----------
    
    collection: 
                a preconstructed S2 collection
              
    start_date: string
                    start date of time series
                    
    end_date: string
                    end date of time series
    EEException: Empty date ranges not supported for the current operation.
    geom: string
          an ogr compatible file with an extent
          
    cloud_filter: int
          cloud pixel percentage in image (default is 60 for s2 cloudless)
          
    Returns
    -------
    
    S2 collection with s2 cloudless and fractional cover included
    
    """
    # cheers google/soilwatch
    # # Global Cloud Masking parameters
    # cld_prb_thresh = 40#; # Cloud probability threshold to mask clouds. 40% is the default value of s2cloudless
    # cloud_filter = 60#; # Threshold on sentinel-2 Metadata field determining whether cloud pixel percentage in image
    # nir_drk_thresh = 0.15#; # A threshold that determines when to consider a dark area a cloud shadow or not
    # cld_prj_dist = 10#; # The distance (in no of pixels) in which to search from detected cloud to find cloud shadows
    # buffer = 50#; # The cloud buffer (in meters) to use around detected cloud pixels to mask additionally
    # mask_res = 60#; # resolution at which to generate and apply the cloud/shadow mask. 60m instead of 10m to speed up
    not_water = ee.Image("JRC/GSW1_2/GlobalSurfaceWater").select('max_extent').eq(0)
    
    # s2 = ee.ImageCollection(collection) \
    # .filterDate(start_date, end_date) \
    # .filterBounds(geom) \
    # .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_filter)
    s2 = collection.filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                   'less_than', cloud_filter)

    # Import and filter s2cloudless.
    s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
      .filterBounds(geom) \
      .filterDate(start_date, end_date)
  
    # To prevent bug the funcs must have the '**' for kwargs
    s2_cl = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
      'primary': s2,
      'secondary': s2_cloudless_col,
      'condition': ee.Filter.equals(**{
          'leftField': 'system:index',
          'rightField': 'system:index'})})).sort('system:time_start')
    
    masked_collection = s2_cl.filter(ee.Filter.notNull(['MEAN_INCIDENCE_AZIMUTH_ANGLE_B3',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B4',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B5',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B6',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B7',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B11',
                                                            'MEAN_INCIDENCE_AZIMUTH_ANGLE_B12',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B3',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B4',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B5',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B6',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B7',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B8A',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B11',
                                                            'MEAN_INCIDENCE_ZENITH_ANGLE_B12',
                                                            'MEAN_SOLAR_AZIMUTH_ANGLE',
                                                            'MEAN_SOLAR_ZENITH_ANGLE'])) \
                            .map(addCloudShadowMask(not_water, 1e4)) \
                            .map(applyCloudShadowMask) \
                            .map(fcover(1e4)) \
                            .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12', 'fcover'])

    return masked_collection


def get_month_ndperc(start_date, end_date, geom, collection="COPERNICUS/S2"):
    
    """
    Make a monthly ndvi 95th percentile composite collection
    
    Parameters
    ----------
              
    start_date: string
                    start date of time series
                    
    end_date: string
                    end date of time series
    
    geom: string
          an ogr compatible file with an extent
          
    collection: string
                the ee image collection  
    
    Returns
    -------

    collection of monthly NDVI images
    
    """
    
    
    roi = extent2poly(geom, filetype='polygon')
    
    imcol = ee.ImageCollection(
            collection).filterDate(start_date, end_date).filterBounds(roi)
    
    months = ee.List.sequence(1, 12)
    
    def _funcmnth(m):
        
        filtered = imcol.filter(ee.Filter.calendarRange(start=m, field='month'))
        
        composite = filtered.reduce(ee.Reducer.percentile([95]))
        
        return composite.normalizedDifference(
                ['B8_p95', 'B4_p95']).rename('NDVI').set('month', m)
    
    composites = ee.ImageCollection.fromImages(months.map(_funcmnth))
    
    return composites

def _feat2dict(lyr, idx, transform=None):
    """
    convert an ogr feat to a dict
    """
    feat = lyr.GetFeature(idx)
    geom = feat.GetGeometryRef()
    if transform != None:
        geom.Transform(transform)
    
    js = geom.ExportToJson()
    geoj = json.loads(js)
    
    # bloody GEE again infuriating
    # prefers lon, lat for points but the opposite for polygons
    # TODO - better fix is required....
    if geoj['type'] == "Point":
        new = [geoj["coordinates"][1], geoj["coordinates"][0]]
        geoj["coordinates"]=new
        
    return geoj

def poly2dictlist(inShp, wgs84=False):
    
    """
    Convert an ogr to a list of json like dicts
    
    Parameters
    ----------
    
    inShp: string
            input OGR compatible polygon
    
    Returns
    -------
    
    List of dicts
    
    """
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    if wgs84 == True:
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
    else:
        transform=None
        
    
    features = np.arange(lyr.GetFeatureCount()).tolist()
    
    # results in the repetiton of first one bug
    # feat = lyr.GetNextFeature() 

    oot = [_feat2dict(lyr, f, transform=transform) for f in features]
    
    return oot
    
    

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
    



def extent2poly(infile, filetype='polygon', outfile=None, polytype="ESRI Shapefile", 
                   geecoord=False):
    
    """
    Get the coordinates of a files extent and return the extent as gee geometry
    
    Parameters
    ----------
    
    infile: string
            input ogr compatible geometry file or gdal raster
            
    filetype: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    outfile: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    polytype: string
            ogr comapatible file type (see gdal/ogr docs) default 'ESRI Shapefile'
            ensure your outfile string has the equiv. e.g. '.shp'

    geecoord: bool
           optionally convert to WGS84 lat,lon
           
    Returns
    -------
    
    a GEE polygon geometry
    
    """
    # ogr read in etc
    if filetype == 'raster':
        ext = _raster_extent(infile)
        
    else:
        # tis a vector
        if isinstance(infile, str):
            vds = ogr.Open(infile)
            lyr = vds.GetLayer()
            ext = lyr.GetExtent()
            srs = lyr.GetSpatialRef()
            
        elif isinstance(infile, gpd.geodataframe.GeoDataFrame):
            ext2 = infile.total_bounds
            ext = [ext2[0], ext2[2], ext2[1], ext2[3]]
            wk = infile.crs.to_wkt()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wk)
    
    # make the linear ring 
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ext[0],ext[2])
    ring.AddPoint(ext[1], ext[2])
    ring.AddPoint(ext[1], ext[3])
    ring.AddPoint(ext[0], ext[3])
    ring.AddPoint(ext[0], ext[2])
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    if geecoord == True:
   
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
        # apply
        poly.Transform(transform)
 
        tproj = wgs84
    else:
        tproj = lyr.GetSpatialRef()
    
    # in case we wish to write it for later....    
    # if outfile != None:
    #     #outfile = infile[:-4]+'extent.shp'
        
    #     out_drv = ogr.GetDriverByName(polytype)
        
    #     # remove output shapefile if it already exists
    #     if os.path.exists(outfile):
    #         out_drv.DeleteDataSource(outfile)
        
    #     # create the output shapefile
    #     ootds = out_drv.CreateDataSource(outfile)
    #     ootlyr = ootds.CreateLayer("extent", tproj, geom_type=ogr.wkbPolygon)
        
    #     # add an ID field
    #     idField = ogr.FieldDefn("id", ogr.OFTInteger)
    #     ootlyr.CreateField(idField)
        
    #     # create the feature and set values
    #     featureDefn = ootlyr.GetLayerDefn()
    #     feature = ogr.Feature(featureDefn)
    #     feature.SetGeometry(poly)
    #     feature.SetField("id", 1)
    #     ootlyr.CreateFeature(feature)
    #     feature = None
        
    #     # Save and close 
    #     ootds.FlushCache()
    #     ootds = None
    
    #     geemap.shp_to_ee(outet.plot_group(file)
    
    # flatten to 2d (done in place)
    poly.FlattenTo2D()
    
    # mind it is a string before this!!!
    ootds = json.loads(poly.ExportToJson())

    return ootds

def s2collection_ts(start_date, end_date, roi, cloud_filter=60, band='NDVI', 
                    agg='month'):
    
    """
    Return an smoothed, aggregated ts collection from Sentinel2
    
    Parameters
    ----------
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
    
    cloud_filter: int
                    s2 cloudless filter
    
    band: string
            the band selected for harmonic regression
            
    agg: string
          period eg week, month
    
    roi: gee geometry 
    
    """
    
    #if isinstance(roi, str):
        
    
    # # unfinished TODO must incorporate this
    # # else assume it is a ee geometry
    # # elif isinstance(roi, ee.geometry.Geometry):
    # elif isinstance(roi, geopandas.geodataframe.GeoDataFrame):
    #     # shapely box
    #     poly = box(*gdf.total_bounds)
    #     poly = json.dumps(mapping(geom)) # not in correct format....
        
   # else:
        #poly = roi
    poly = roi
    
    # this needs updated yearly - think of something better
    years = ee.Dictionary({'2016': 'COPERNICUS/S2',
                                '2017': 'COPERNICUS/S2',
                                '2018': 'COPERNICUS/S2',
                                '2019': 'COPERNICUS/S2_SR',
                                '2020': 'COPERNICUS/S2_SR',
                                '2021': 'COPERNICUS/S2_SR',
                                '2022': 'COPERNICUS/S2_SR'})

    # date range dict
    dts = {'start': start_date, 'end': end_date}
    date_range_temp = ee.Dictionary(dts)
    year = start_date[0:4]

    # Load the Sentinel-2 collection for the time period and area requested
    s2_cl = loadImageCollection(ee.String(years.get(year)).getInfo(), 
                                 date_range_temp, poly)

    S2cld = s2cloudless(s2_cl, start_date, end_date, poly,
                     cloud_filter=cloud_filter)
    
    # pointless repetition (should likely go in s2cloudless)
    S2 = (ee.ImageCollection(S2cld)
         .spectralIndices([band])) # should more be calculated?
     
    S2final = harmonic_regress(S2, dependent=band, harmonics=3) 
     
    ts2 = wxee.TimeSeries(S2final).select('fitted')
    ts_fl2 = ts2.aggregate_time(frequency=agg, reducer=ee.Reducer.median())
    
    return ts_fl2
    
    

def zonal_tseries(inShp, start_date, end_date, bandnm='NDVI',
                  attribute='id', scale=20):
    
    
    """
    Zonal Time series for a feature collection 
    
    Parameters
    ----------
    
    inShp: string
            path to input shapefile
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
             
    bandnm: string
             the bandname of choice that exists or has been created in 
             the image collection  e.g. B1 or NDVI
            
    attribute: string
                the attribute for filtering (required for GEE)
    
    scale: int
            pixel size in metres
                
    Returns
    -------
    
    pandas dataframe
    
    Notes
    -----
    
    Unlike the other tseries functions here, this operates server side, meaning
    the bottleneck is in the download/conversion to dataframe.
    
    """
    if isinstance(inShp, str):
        
        shp = geemap.shp_to_ee(inShp)
        
    elif isinstance(inShp, gpd.geodataframe.GeoDataFrame):
        
        shp = geemap.gdf_to_ee(inShp) 
    
    elif isinstance(inShp, ee.featurecollection.FeatureCollection):
        
        shp = inShp
    
    poly = shp.geometry().bounds()
    
    if bandnm == 'GEOS3':
        
        bs_masked, _, _ = baresoil_collection(poly, start_date=start_date, 
                                              end_date=end_date, 
                                              band_list=['fcover'])
        
        dep = 'fcover'
        # doesn't work without this stage - must find out why....
        bsfinal = harmonic_regress(bs_masked, dependent=dep, harmonics=3)
        #No longer using pandas as no need
        ts = wxee.TimeSeries(bsfinal).select(['GEOS3'])
        
        # this must be maximum to retain the baresoil binary values
        # this does render the fcover pointless as you need the average of that
        collection = ts.aggregate_time(frequency='month', reducer=ee.Reducer.max())
        
        sel = 'GEOS3'
        # for later fixing
        stat = 'max'
        
    else:
        collection = s2collection_ts(start_date, end_date, poly, cloud_filter=60, 
                                 band=bandnm, agg='month')
        sel = 'fitted'
        # for later fixing
        stat = 'median'
    
    # select the band and perform a spatial agg
    # GEE makes things ugly/hard to read
    # I have nested rather than using lambda here and it works
    def _imfunc(image): 
        
        def wrapf(f):
            return f.set('imageId', image.id())
        
        return (image.select(sel)
        .reduceRegions(
          collection=shp.select([attribute]),
          reducer=ee.Reducer.mean(),
          scale=scale
        )
        .filter(ee.Filter.neq('mean', None))
        .map(wrapf))

    # now map the above over the collection
    # we have a triplet of attributes, which can be rearranged to a table
    triplets = collection.map(_imfunc).flatten()

    def _fmt(table, row_id, col_id):
      """
      arrange the image stat values into a table of specified order
      """ 
      def _funcfeat(feature):
              feature = ee.Feature(feature)
              return [feature.get(col_id), feature.get('mean')]
          
      def _rowfunc(row):
          
          values = ee.List(row.get('matches')).map(_funcfeat)
          
          return row.select([row_id]).set(ee.Dictionary(values.flatten()))
    
      rows = table.distinct(row_id)
      # Join the table to the unique IDs to get a collection in which
      # each feature stores a common row ID.
      joined = ee.Join.saveAll('matches').apply(primary=rows,
                              secondary=table,
                              condition=ee.Filter.equals(leftField=row_id, 
                                                         rightField=row_id))
      
      t_oot = joined.map(_rowfunc)
      
      return t_oot 
    
    # (of course attribute will appear as a column in the df)
    #             t          rowid       colid
    table = _fmt(triplets, attribute, 'imageId')
    
    print('downloading from gee')
    df = geemap.ee_to_pandas(table)
    
    # Now we need to sort the columns by date which are not in order
    df.sort_index(axis=1, inplace=True)
    
    # right, at this point we need to fix the variable length and fill
    # the gaps
    
    
    # format the date/veg ind columns
    names = [c.split(sep='_')[0][0:8] for c in df.columns]
    

    names = names[:-1] # get rid of attribute - 
    
    bname = bandnm[0].lower()+'-'
    
    # have filled the gaps in the previous hacky manner
    dnames = [_dodate2(n) for n in names]
    dnames.append(attribute)
    df.columns = dnames
    
    # stat is determined by input collection - if bs then max, otherwise median
    dft =_fix_ts(df, start_date, end_date, attribute, bname, agg='M', stat=stat)
    
    
    #TODO open gdf & merge, then return and/or write
    # do we merge it here or do this outside func to stop opening & closing
    # multiple files?
    # gdf = gpd.read_file(inshp)

    # newdf = pd.concat([gdf, df], axis=1)
    
    # if outfile != None:
    #     newdf.to_file(outfile) # todo? driver='GPKG', layer='name')
    
    # return newdf
    
    # for a larger table this may be required
    # task = ee.batch.Export.table.toDrive(table, folder='earthengine', 
    #                                      fileNamePrefix='zonaltst',
    #                                      fileFormat='csv')

    # task.start()
    #TODO
    # while task.status()['state'] == 'RUNNING':
    #     print('')
    
    # OR
    
    # geemap.ee_export_vector_to_drive(
    # fc, description="europe", fileFormat='SHP', folder="export")


    return dft

def _fix_ts(df, start_date, end_date, attribute, bname, agg='M', stat='median'):
    
    # another dire hack to get the job done
    
    dft = df.transpose()
    dft = dft.drop(dft[dft.index == 'idx'].index)
    dft['Date'] = pd.to_datetime(dft.index)
    dft = dft.set_index(dft['Date'])
    dft.drop(columns='Date', inplace=True)
    # TODO add the agg(M) and stat(median) params further up 
    # as different depending on bs or ndvi
    dft = _fixgaps(dft, agg, stat, start_date, end_date)
    
    dft = dft.transpose()
    
    dft.columns = dft.columns.strftime('%Y-%m-%d')
    
    dft.columns = [bname+c[2:] for c in dft.columns]
    
    dft[attribute] = df[attribute]
    dft.index.rename('index', inplace=True)
    
    return dft
    
def _dodate(n, bandname):
    # awful
    oot = bandname + n[2:4] +'-'+n[4:6]+'-'+n[6:8]
    return oot

def _dodate2(n):
    #ditto 
    oot = n[0:4] +'-'+n[4:6]+'-'+n[6:8]
    return oot

def points_to_pixel(gdf, espgin='epsg:27700', espgout='epsg:4326'):
    """
    convert some points from one proj to the other and return pixel coords
    and lon lats for some underlying raster
    
    returns both to give choice between numpy/pandas or xarray
    
    """
    transformer = Transformer.from_crs(espgin, espgout, always_xy=True) 
    xin = gdf.POINT_X
    yin = gdf.POINT_Y
    
    # better than the old pyproj way
    points = list(zip(xin, yin))
    
    # output order is lon, lat
    coords_oot = np.array(list(transformer.itransform(points)))
    
    # for readability only
    lats = coords_oot[:,1]
    lons = coords_oot[:,0]
    
    
    return  lons, lats


# conver S2 name to date
def _conv_date(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[0][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot    

def _conv_dateS1(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[5][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot  

def _conv_date3(entry, name):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.replace(name+'-', '20') 
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot   

def simplify(fc):
    def feature2dict(f):
            id = f['id']
            out = f['properties']
            out.update(id=id)
            return out
    out = [feature2dict(x) for x in fc['features']]
    return out

def harmonic_regress(collection, dependent='NDVI', harmonics=3):
    
    """
    Fit a harmonic regression to smooth a time series using GEE
    
    Parameters
    ----------
    
    collection: gee image collection
                e.g. S2 with NDVI/fcover calculated
    
    dependent: string
               the dependent variable usually a veg index 
                
    harmonics: int
               harmonic components
    """
    
    # Credit to Joao Otavio Nascimento Firigato for this implementation
    # I merely made it a function for any collection
    
    
    harmonicFrequencies = list(range(1, harmonics+1))
    
    def getNames(base, lst_freq):
        name_lst = []
        for i in lst_freq:
            name_lst.append(ee.String(base + str(i)))
        return name_lst
    
    cosNames = getNames('cos_', harmonicFrequencies)
    sinNames = getNames('sin_', harmonicFrequencies)
    independents = ee.List(['constant','t']).cat(cosNames).cat(sinNames)
    
    def addConstant(image):
        return image.addBands(ee.Image(1))
    
    def addTime(image):
        date = ee.Date(image.get('system:time_start'))
        # may need to alter this...
        years = date.difference(ee.Date('1970-01-01'), 'year')
        timeRadians = ee.Image(years.multiply(2 * math.pi))
        return image.addBands(timeRadians.rename('t').float())
    
    def addHarmonics(image):
        frequencies = ee.Image.constant(harmonicFrequencies)
        time = ee.Image(image).select('t')
        cosines = time.multiply(frequencies).cos().rename(cosNames)
        sines = time.multiply(frequencies).sin().rename(sinNames)
        return image.addBands(cosines).addBands(sines)
    
    # used modis but now any given collection
    harmonicC = collection.map(addConstant).map(addTime).map(addHarmonics)
    
    harmonicTrend = harmonicC.select(independents.add(dependent)).reduce(ee.Reducer.linearRegression(independents.length(), 1))
    
    harmonicTrendCoefficients = harmonicTrend.select('coefficients').arrayProject([0]).arrayFlatten([independents])
    
    fittedHarmonic = harmonicC.map(
        lambda image : image.addBands(image.select(
            independents).multiply(harmonicTrendCoefficients).reduce('sum').rename('fitted')))
    
    # so the sum of the 1st 3 harmonic terms is the output ts
    
    return fittedHarmonic
    


def _bs_tseries(geometry,  start_date='2021-01-01', end_date='2021-12-31', dist=20, 
               stat='mean', para=False, bandList=['GEOS3', 'fitted'],
               agg='week'):
    
    """
    Bare soil Time series from a single coordinate/polygon using gee
    
    Parameters
    ----------
    
    geometry: geodataframe
                a geodataframe of a single geometry

    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m

    agg: string
            aggregate to... 'week', 'month'
              
    """
    # joblib hack - this is goona need to run in gee directly i think
    if para == True:
        ee.Initialize()
        
    #geometry = geemap.gdf_to_ee(geometry)
    
    # has to reside in here in order for para execution
    # TODO scale needs to be made and explcit param above 
    def _reduce_region(image):
        # cheers geeextract!   
        """Spatial aggregation function for a single image and a polygon feature"""
        stat_dict = image.reduceRegion(fun, geometry, 20);
        # FEature needs to be rebuilt because the backend doesn't accept to map
        # functions that return dictionaries
        return ee.Feature(None, stat_dict)
    
    # we only need the collection with the masks as a layer
    bs_masked, _, _ = baresoil_collection(geometry, start_date=start_date, 
                                          end_date=end_date, 
                                          band_list=['fcover'])
    
    # This produces smooth curves are they too similar or not from
    # one yr to the next
    # worth remembering it is cover rather than ndvi....
    
    #TODO Do we really need fcover particulalrly? Is this slowing it up?
    # Tempting to remove it as using the max stat is making it pointless 
    dep = 'fcover'
    bsfinal = harmonic_regress(bs_masked, dependent=dep, harmonics=3)
    #No longer using pandas as no need
    ts = wxee.TimeSeries(bsfinal).select(['fitted', 'GEOS3'])
    
    # this must be maximum to retain the baresoil binary values
    # this does render the fcover pointless as you need the average of that
    ts_fl = ts.aggregate_time(frequency=agg, reducer=ee.Reducer.max())
    
    #dates = ts_fl.aggregate_array("system:time_start")
    #dts = dates.getInfo() # not in readable format
    
                          
    if geometry['type'] == 'Polygon':
        
        
        # have to choose median to get both the mask(the mask is binary of course)
        # and the fcover 
        # mean gives the proportion of polygon covered so maybe reconsider the
        # above
        if stat == 'mean':
            fun = ee.Reducer.mean()
        elif stat == 'median':
            fun = ee.Reducer.median()
        elif stat == 'max':
            fun = ee.Reducer.max()
        elif stat == 'min':
            fun = ee.Reducer.min()
        elif stat == 'perc':
            # for now as don't think there is equiv
            fun = ee.Reducer.mean()
        else:
            raise ValueError('Must be one of mean, median, max, or min')
        # ideal would be mode but this is not suitable for fcover
        # don't want to calculate this twice after all as it will slow things

        geomee = ee.Geometry.Polygon(geometry['coordinates'])
        
        # 23secs for monthly 4 yrs
        # 27 secs for weekly 
        # 30 secs for all dates 4 yrs
        #start = timeit.default_timer()
        bsList = ts_fl.filterBounds(geomee).map(_reduce_region).getInfo()
        #stop = timeit.default_timer()
        #print((stop - start)) 
        
        
        bsList = simplify(bsList)
        
        df = pd.DataFrame(bsList)
        
        
    elif geometry['type'] == 'Point':
    
        # a point
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # give the point a distance to ensure we are on a pixel
        bsList = ts_fl.filterBounds(geomee).getRegion(geomee, dist).getInfo()
    
        # the headings of the data
        cols = bsList[0]
        
        # the rest
        rem=bsList[1:len(bsList)]
        
        # now a big df to reduce somewhat
        df = pd.DataFrame(data=rem, columns=cols)
    else:
        raise ValueError('geom must be either Polygon or Point')

    # get a proper date
    df['Date'] = df['id'].apply(_conv_date)
    
    # May change to this for merging below
    df = df.set_index(df['Date'])

    # should be this but....
    bs = df[bandList]    
    #TODO fix issue in merge in main func to do with shape
    # finaldf = pd.DataFrame(datalist)
    # when creating the final df it complains about the shape being (n,n,n)
    # so having to export a series - have tried list of series and didn't work
    #bs = pd.Series(df['GEOS3'])
    #fcover = pd.Series(df['fitted'])
    
    # As we used the spatial mean earlier the bare soil needs rounded up 
    # to be binary again
    # stop the stupid warning
      # default='warn'
    # this seems to b

    bs['GEOS3'].fillna(0, inplace=True)
    
    
    if stat == 'max':
        bs['GEOS3'] = np.ceil(bs['GEOS3']).astype(int)
    

    pdagg = agg[0].capitalize()
    
    # create an even ts as sometimes all wks/mnths do not appear in localities
    # Always take the temporal max. To get the spatial proportion of a field
    # take the spatial average of the binary values
    bs = _fixgaps(bs, pdagg, 'max', start_date, end_date)
    
    
    return bs.transpose()


def _fixgaps(d, agg, stat, start_date, end_date):
    
    # Issue is with S2, where there don't appear to be imgs for certain locations
    # at certain times
    # TODO this is still problemtatic - don't ken why it occurs
    # fixes gaps within but not at the beggining of ts. 
    # it may be the 
    
    
    # ts from GEE often end up with months missing and 2 dates in one month 
    # instead. Hence this to sort it out
    if stat == 'max':
        d = d.resample(rule=agg).max()
    if stat == 'perc':
        # avoid potential outliers - not relevant to SAR....
        d = d.resample(rule=agg).quantile(.95)
    if stat == 'mean':
        d = d.resample(rule=agg).mean()
    if stat == 'median':
        d = d.resample(rule=agg).median()
    
    # change to end of month to be inline with pd
    d.index = d.index+pd.offsets.MonthEnd(0)
    
    # interpolate the nan gaps
    d = d.interpolate()
    
    # Hack......
    chk_rng = pd.DataFrame(pd.date_range(start_date, end_date, freq='M'),
                           columns=['Date'])
    if chk_rng['Date'].size > d.index.size:
        
        d = chk_rng.merge(d, how='left', left_on='Date', right_on=d.index)
        d.index = d.Date
        d = d.drop(columns=['Date'])
        # all a terrible fix/hack
        d.fillna(0, inplace=True)
        d.fitted.backfill(inplace=True)
        d.fitted.ffill(inplace=True)
        
    
    return d 

def bs_ts(inshp, reproj=False, start_date='2021-01-01', end_date='2021-12-31', 
          dist=20, stat='median', para=False, 
          agg='week', outfile=None, nt=-1):
    
    
    """
    Bare soil Time series from polygons/points using gee
    
    Parameters
    ----------
    
    geometry: list


    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
    
    outfile: string
           the output shapefile if required

    agg: string
            aggregate to... 'week', 'month'
              
    """
    # this seems to causing issues w/ cs polys
    geom = poly2dictlist(inshp, wgs84=reproj)
    
    idx = np.arange(0, len(geom))
    
    gdf = gpd.read_file(inshp)
    
    #idx = np.arange(0, gdf.shape[0])
    
    # silly gpd issue
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    datalist = Parallel(n_jobs=nt, verbose=2)(delayed(_bs_tseries)(
                    geom[p],#gdf.iloc[[p]],
                    start_date=start_date,
                    end_date=end_date,
                    stat=stat,
                    para=True,
                    agg=agg) for p in idx)
    
    # The origin of the issue is there are multiple dates in the same month
    # which should in theory not be happening in the previous aggs...

    # it has got messy
    # this is incredibly slow on even small data

    #TODO There must be a more elegant/efficient way
    
    listarrs = [d.to_numpy().flatten() for d in datalist]
    finaldf = pd.DataFrame(np.vstack(listarrs))
    del listarrs
    
    colstmp = []
    bandlist = ['g', 'f']
    # this seems inefficient, but gets the job done as simply chucking the list
    # of dfs as pd.Dataframe does not work
    # if agg == 'month':
    #     times = datalist[0].columns.strftime("%y-%m").tolist() #* (len(bandlist)+1)
    # else:
    # in line w/ s2_ts
    times = datalist[0].columns.strftime("%y-%m-%d").tolist()
    for b in bandlist:
        tmp = [b+"-"+ t for t in times]
        colstmp.append(tmp)
    # apperently quickest
    colsfin = list(chain.from_iterable(colstmp))
    finaldf.columns = colsfin

    # idx must be unique so make it the gdf one
    finaldf.index = gdf.index 
    newdf = pd.concat([gdf, finaldf], axis=1)
    
    if outfile != None:
        newdf.to_file(outfile) # todo? driver='GPKG', layer='name')
    
    return newdf


def _s2_tseries(geometry, start_date='2016-01-01',
               end_date='2016-12-31', dist=20, 
               stat='median', cloud_filter=60, bandlist=['NDVI'], para=False,
               agg='month'):
    
    """
    S2 Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    geometry: list
                either [lon,lat] fot a point, or a set of coordinates for a 
                polygon
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    cloud_filter: int
             whether to mask cloud
    
    bandlist: List of string(s)
             the bands to use as time series 

    agg: string
            aggregate to... 'week', 'month'
              
    """
    # joblib hack - this is gonna need to run in gee directly i think
    if para == True:
        ee.Initialize()
    
    # has to reside in here in order for para execution
    def _reduce_region(image):
        # cheers geeextract!   
        """Spatial aggregation function for a single image and a polygon feature"""
        stat_dict = image.reduceRegion(fun, geometry, 30);
        # FEature needs to be rebuilt because the backend doesn't accept to map
        # functions that return dictionaries
        return ee.Feature(None, stat_dict)
    
    #TODO this results in the loss of NDVI.....
    #eemont may juts slow things down - may just add NDVI in masks module
    
    years = ee.Dictionary({'2016': 'COPERNICUS/S2',
                               '2017': 'COPERNICUS/S2',
                               '2018': 'COPERNICUS/S2',
                               '2019': 'COPERNICUS/S2_SR',
                               '2020': 'COPERNICUS/S2_SR',
                               '2021': 'COPERNICUS/S2_SR',
                               '2022': 'COPERNICUS/S2_SR'})

    # date range dict
    dts = {'start': start_date, 'end': end_date}
    date_range_temp = ee.Dictionary(dts)
    year = start_date[0:4]

    # Load the Sentinel-2 collection for the time period and area requested
    s2_cl = loadImageCollection(ee.String(years.get(year)).getInfo(), 
                                date_range_temp, geometry)
    
    S2cld = s2cloudless(s2_cl, start_date, end_date, geometry,
                    cloud_filter=cloud_filter)
    # pointless repetition (should likely go in s2cloudless)
    S2 = (ee.ImageCollection(S2cld)
        .spectralIndices(['NDVI']))
    
    # how does one map over a list in GEE
    # def hrfunc(collection, band):
    #     harmonic_regress(S2, dependent=band, harmonics=3)
    
    
    S2final = harmonic_regress(S2, dependent='NDVI', harmonics=3) 
    
    #No longer using pandas as no need
    ts = wxee.TimeSeries(S2final).select('fitted')
    ts_fl = ts.aggregate_time(frequency=agg, reducer=ee.Reducer.median())
                     
    
    if geometry['type'] == 'Polygon':
        
        # cheers geeextract folks 
        if stat == 'mean':
            fun = ee.Reducer.mean()
        elif stat == 'median':
            fun = ee.Reducer.median()
        elif stat == 'max':
            fun = ee.Reducer.max()
        elif stat == 'min':
            fun = ee.Reducer.min()
        elif stat == 'perc':
            # for now as don't think there is equiv
            fun = ee.Reducer.mean()
        else:
            raise ValueError('Must be one of mean, median, max, or min')
        geomee = ee.Geometry.Polygon(geometry['coordinates'])
        
        S2List = ts_fl.filterBounds(geomee).map(_reduce_region).getInfo()
        S2List = simplify(S2List)
        
        df = pd.DataFrame(S2List)
        
        
    elif geometry['type'] == 'Point':
    
        # a point
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # give the point a distance to ensure we are on a pixel
        s2list = ts_fl.filterBounds(geomee).getRegion(geomee, dist).getInfo()
    
        # the headings of the data
        cols = s2list[0]
        
        # the rest
        rem=s2list[1:len(s2list)]
        
        # now a big df to reduce somewhat
        df = pd.DataFrame(data=rem, columns=cols)
    else:
        raise ValueError('geom must be either Polygon or Point')

    # get a proper date
    df['Date'] = df['id'].apply(_conv_date)
    
    # May change to this for merging below
    df = df.set_index(df['Date'])
    
    nd = df['fitted']  # change this in the harmonic code....
    
    pdagg = agg[0].capitalize()
    
    nd = _fixgaps(nd, pdagg, stat, start_date, end_date)
    
    return nd.transpose()
    

# A quick/dirty answer but not an efficient one - this took almost 2mins....

def S2_ts(inshp, reproj=False,
          start_date='2016-01-01', end_date='2016-12-31', dist=20, 
          stat='median', cloud_filter=60, 
          bandlist=['NDVI'], para=False, outfile=None, nt=-1,
               agg='month'):
    
    
    """
    Monthly time series from a point shapefile - cloud is masked 
    with S2 cloudless and collection is derived by year
    
    Parameters
    ----------
             
    inshp: string
                a shapefile to join the results to 
    
    reproj: bool
                whether to reproject to wgs84, lat/lon
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
    
    bandList: List of str
             the bands to use as time series 
             
    cloud_filter: int
             the acceptable cloudiness per pixel in addition to prev arg
    
    outfile: string
           the output shapefile if required

    agg: string
            aggregate to... 'week', 'month'
             
             
    Returns
    -------
         
    geopandas dataframe
    
    Notes
    -----
    
    This spreads the point queries client side, meaning the bottleneck is the 
    of threads you have. This is maybe 'evened out' by returning the full dataframe 
    quicker than dowloading it all from the server side and loading back into memory
    
    """
    # possible to access via gpd & the usual shapely fare....
    # geom = gdf['geometry'][0]
    # geom.exterior.coords.xy
    # but will lead to issues....
    
    geom = poly2dictlist(inshp, wgs84=reproj)
    
    idx = np.arange(0, len(geom))
    
    gdf = gpd.read_file(inshp)
    
    # silly gpd issue
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    datalist = Parallel(n_jobs=nt, verbose=2)(delayed(_s2_tseries)(
                    geom[p],
                    start_date=start_date,
                    end_date=end_date,
                    stat=stat,
                    cloud_filter=cloud_filter,
                    bandlist=bandlist,
                    para=True,
                    agg=agg) for p in idx) 
    
    #if len(bandlist) > 1:
        
    #TODO There must be a more elegant/efficient way
    listarrs = [d.to_numpy().flatten() for d in datalist]
    finaldf = pd.DataFrame(np.vstack(listarrs))
    del listarrs
    
    colstmp = []
    # this seems inefficient
    # if agg == 'month':
    #     times = datalist[0].columns.strftime("%y-%m").tolist() #* (len(bandlist)+1)
    # else:
    times = datalist[0].index.strftime("%y-%m-%d").tolist()
    for b in bandlist:
        tmp = [b[0].lower()+"-"+ t for t in times]
        colstmp.append(tmp)
    # apperently quickest
    colsfin = list(chain.from_iterable(colstmp))

    finaldf.columns = colsfin

        
    #else:
    
        # finaldf = pd.DataFrame(datalist)
        
        # # Unfortunately monthly aggs do not always occur via GEE it seems
        # # can result in duplicate values eg 2 aprils then geopandas won't
        # # write the file
        # # if agg == 'month':
        # #     finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
        # #     finaldf.columns = ["n-"+c for c in finaldf.columns]
        # #else:
        # finaldf.columns = finaldf.columns.strftime("%y-%m-%d").to_list()
        # finaldf.columns = ["n-"+c for c in finaldf.columns]

    # for some reason merge no longer working due to na error when there is none
    # hence concat. If they are in correct order no reason to worry as
    # no index is present in the ndvi df anyway. 
    #newdf = pd.merge(gdf, finaldf, on=gdf.index)
    
    # idx must be unique so make it the gdf one
    finaldf.index = gdf.index 
    newdf = pd.concat([gdf, finaldf], axis=1)
    
    if outfile != None:
        newdf.to_file(outfile) # todo? driver='GPKG', layer='name')
    
    return newdf

def plot_group(df, group, index, name,  year=None, title=None, fill=False,
               freq='M', plotstat='mean'):
    
    """
    Plot time series per poly or point eg for S2 ndvi, met var, S1
    If the the no of entries > 20, then the average/median with min/max
    shading will be plotted
    
    Parameters
    ----------
    
    df: byte                     
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: list
            the index of interest
            
    name: string
            the name of interest must include the dash before the name
            eg for ndvi by day or week it is 'n-'
            
    legend_col: string
            the column that legend lines will be labeled by
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)
            
    title: string
            plot title

    fill: bool
            fill the area between plot lines 
            (this a bit imprecise in some areas)
    
    plotstat: string
            the line to plot when the dataset is too big (> 20 entries)
            either 'mean' or 'median'
    
    """
    
    # Quick dirty time series plotting
    
    
    #TODO potentially drop this as could be donw outside func
    #Not sure why I have stuck with this
    sqr = df.loc[df[group].isin(index)]
    
    
    yrcols = [y for y in sqr.columns if name in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y[0:4]]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq=freq)
    else:
        
        # set the dtrange......
        # this code is dire  due to way i wrote other stuff from gee plus
        # ad hoc changes as I go
        #TODO alter this crap
        # if freq == 'M':
        #     startd = yrcols[0][-5:]
        #     startd = '20'+startd+'-01'
        #     # this seems stupid
        #     endd = yrcols[-1:][0][-5:]
        #     endd = '20'+endd[0:2]+'-12-31'
        #     dtrange = pd.date_range(start=startd, end=endd, freq=freq)
            
        #else:
        startd = yrcols[0][-8:]
        startd = '20'+startd
        endd = yrcols[-1:][0][-8:]
        endd = '20'+endd
        #dtrange = pd.date_range(start=startd, end=endd, freq=freq)

            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    
    new = ndplotvals.transpose()
    
    #if freq != 'M':
    new['Date'] = new.index
    new['Date'] = new['Date'].str.replace(name,'20')
    new['Date'] = new['Date'].str.replace('_','0')
    new['Date'] = pd.to_datetime(new['Date'])
    #else:
    #new['Date'] = dtrange
    
    
    new = new.set_index('Date')
    # what is this doing....
    #new.columns=[name]
    # to add the index as the legend
    #new['mean'] = new.mean()
    if plotstat == 'mean':
        mn = new.mean(axis=1)
    elif plotstat == 'median':
        mn = new.median(axis=1)
    
    if len(new.columns) < 20:
        ax = new.plot.line(title=title)
    else:
        ax = mn.plot.line(title=title, label=plotstat)
    
    if fill == True:
         minr = new.min(axis=1, numeric_only=True)
         maxr = new.max(axis=1, numeric_only=True)
         stdr = new.std(axis=1, numeric_only=True)
         ax.fill_between(new.index, minr, maxr, alpha=0.3, label='Min/Max')
         # plot std around each point
         ax.fill_between(new.index, (mn-2*stdr), 
                         (mn+2*stdr), color='r', alpha=0.1, label='Stdev')
    if year is None:
        # does not appear to work....
        # TODO this could be adapted to highlight the phenostages with their
        # labels
        xpos = [pd.to_datetime(startd), pd.to_datetime(endd)]
        for xc in xpos:
            ax.axvline(x=xc, color='k', linestyle='-')
            
    

    if len(new.columns) < 20:
        ax.legend(labels=sqr[group].to_list())
    else:
        ax.legend()
    plt.show()

def plot_crop(df, group, index, name, crop="SP BA", year=None, title=None,
              fill=False, freq='M'):
    
    """
    Plot time series per poly or point eg for S2 ndvi, S1, or met var
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: list
            the index of interest
            
    name: string
            the name of interest
            
    crop: string
            the crop of interest
            
    legend_col: string
            the column that legend lines will be labeled by
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)
            
    title: string
            plot title

    fill: bool
            fill the area between plot lines 
            (this a bit imprecise in some areas)
    
    """
    
    # Quick dirty time series plotting of crops with corresponding stages
    

    #TODO potentially drop this as could be donw outside func
    sqr = df.loc[df[group].isin(index)]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y[0:4]]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq=freq)
    else:
        
        # set the dtrange......
        # this code is dire  due to way i wrote other stuff from gee plus
        # ad hoc changes as I go
        #TODO alter this crap
        # if freq == 'M':
        #     startd = yrcols[0][-5:]
        #     startd = '20'+startd+'-01'
        #     # this seems stupid
        #     endd = yrcols[-1:][0][-5:]
        #     endd = '20'+endd[0:2]+'-12-31'
        #     dtrange = pd.date_range(start=startd, end=endd, freq=freq)
            
        #else:
        startd = yrcols[0][-8:]
        startd = '20'+startd
        endd = yrcols[-1:][0][-8:]
        endd = '20'+endd
        #dtrange = pd.date_range(start=startd, end=endd, freq=freq)

            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    
    new = ndplotvals.transpose()
    
    #if freq != 'M':
    new['Date'] = new.index
    new['Date'] = new['Date'].str.replace(name,'20')
    new['Date'] = new['Date'].str.replace('_','0')
    new['Date'] = pd.to_datetime(new['Date'])
    #else:
    #new['Date'] = dtrange
    
    
    new = new.set_index('Date')
    
    #new.columns=[name]
    
    # it is not scaled to put lines between months....
    # but the line doesn't plot so sod this
#    days = pd.date_range(start='20'+year+'-01-01',
#                                end='20'+year+'-12-31',
#                                freq='D')
#    
#    new = new.reindex(days)
#   and before you try it - dropna just returns the axis to monthly.
    
#    # to add the index as the legend
    ax = new.plot.line(title=title)
    

    

    if fill == True:
         minr = new.min(axis=1, numeric_only=True)
         maxr = new.max(axis=1, numeric_only=True)
         ax.fill_between(new.index, minr, maxr, alpha=0.1, label='min')
    
    if year is None:
        # TODO this could be adapted to highlight the phenostages with their
        # labels
        xpos = [pd.to_datetime(startd), pd.to_datetime(endd)]
        for xc in xpos:
            ax.axvline(x=xc, color='k', linestyle='--', label='pish')
        ax.axvspan(xpos[0], xpos[1], facecolor='gray', alpha=0.2)

    ax.legend(labels=sqr[group].to_list(), loc='center left', 
              bbox_to_anchor=(1.0, 0.5))
   
    
    yr = "20"+year
    # The crop dicts -approx patterns of crop growth....
    # this is a hacky mess for now
    # Problem is axis is monthly from previous
    # so changing to days or weeks doesn't work, including half months
    crop_dict = {"SP BA": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
                           pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
                           pd.to_datetime(yr+"-09-01")],
                 "SP BE": [],
                 "SP Oats": [],
                 "SP WH": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
                           pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
                           pd.to_datetime(yr+"-09-01")],
                 "W Rye": [],
                 "W Spelt": [],
                 "W WH": []}
    
    # these are placeholders and NOT correct!!!
    # mind the list much be same length otherwise zip will stop at end of shortest
    # also the last entry is of course not between lines
    crop_seq = {"SP BA": [' Em', ' Stem' ,' Infl',
                           ' Flr>\n Gr>\n Sen',
                           ""],
                 "SP BE": [],
                 "SP Oats": [],
                 "SP WH": [' Em', ' Stem' ,' Infl',
                           ' Flr>\n Gr> \nSen',
                           ""],
                 "W Rye": [],
                 "W Spelt": [],
                 "W WH": []}
    
    # rainbow
    clrs = cm.rainbow(np.linspace(0, 1, len(crop_dict[crop])))
    
    #for the text 
    style = dict(size=10, color='black')
    
    
    #useful stuff here
    #https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html
    
    # this is also a mess
    # Gen the min of mins for the position of text on y axis
    minr = new.min(axis=1, numeric_only=True)
    btm = minr.min() - (minr.min() / 2)
    for xc, nm in zip(crop_dict[crop], crop_seq[crop]):
        ax.axvline(x=xc, color='k', linestyle='--')
        ax.text(xc, minr.min(), " "+nm, ha='left', **style)
        # can't get this to work as yet
#        ax.annotate(nm, xy=(xc, 0.1),
#                    xycoords='data', bbox=dict(boxstyle="round", fc="none", ec="gray"),
#                    xytext=(10, -40), textcoords='offset points', ha='center',
#                    arrowprops=dict(arrowstyle="->"), 
#                    annotation_clip=False)
   # theend = len(crop_dict[crop])-1
    
    for idx, c in enumerate(clrs):
        if idx == len(clrs)-1:
            break
        ax.axvspan(crop_dict[crop][idx], crop_dict[crop][idx+1], 
                   facecolor=c, alpha=0.1)
        
    
    
    # so if we take the mid date point we can put the label in the middle
 
    # hmm this does not do what I had hoped, it's not in the middle....
#    midtime = xpos[0] + (xpos[1] - xpos[0])/2
    # can cheat by putting a space at the start!!!!
    #ax.text(midtime, 0.5, " crop", ha='left', **style)
    plt.show()
    
def _doit(new, name):
    new['Date'] = new.index
    new['Date'] = new['Date'].str.replace(name,'20')
    new['Date'] = new['Date'].str.replace('_','0')
    new['Date'] = pd.to_datetime(new['Date'])
    return new

def plot_soil_fcover(df, group, index, name, crop="SP BA", year=None, title=None,
              fill=False, freq='M'):
    
    """
    Plot time series per poly or point eg for S2 ndvi, S1, or met var
    
    Parameters
    ----------
    
    df: byte
        input pandas/geopandas df
    
    group: string
          the attribute to group by
          
    index: list
            the index of interest
            
    name: string
            the name of interest
            
    crop: string
            the crop of interest
           
    legend_col: string
            the column that legend lines will be labeled by
    
    year: string
            the year to summarise e.g. '16' for 2016 (optional)
            
    title: string
            plot title

    fill: bool
            fill the area between plot lines 
            (this a bit imprecise in some areas)
    
    """
    
    # Quick dirty time series plotting of bare soil & fcover
    #TODO need tick mark bare soil to match color of fcover line, or be
    # annotated

    #write plot only when bare soil occured and the coincident fcover
    b = df.filter(regex='g-')
    n = df.filter(regex='f-')

    # subset
    sub = b[b.values==1]
    idx = sub.index.tolist()
    bsub = b[b.values==1]
    nsub = n.iloc[sub.index.tolist()]
    
    yrcols = [y for y in bsub.columns if 'g-' in y]
    
    slnew = bsub.transpose()
    new = nsub.transpose()
    
    new = _doit(new, "f-")
    slnew = _doit(slnew, "g-")

            
    # TODO - this is crap really needs replaced....
    # ndplotvals = sqr[yrcols]
    # soilplotvals = sqr[soilcols]
    
    # slnew = soilplotvals.transpose()
    # new = ndplotvals.transpose()
    
        
    #if freq != 'M':
    new = _doit(new, "f-")
    slnew = _doit(slnew, "g-")
        
    # else:
    #     new['Date'] = dtrange
    #     slnew['Date'] = dtrange
    
    new = new.set_index('Date')
    slnew = slnew.set_index('Date')
    slbare = slnew[slnew.values>0]
    
    soildates = slbare.index
    
    #new.columns=[name, name]
    
    # it is not scaled to put lines between months....
    # but the line doesn't plot so sod this
#    days = pd.date_range(start='20'+year+'-01-01',
#                                end='20'+year+'-12-31',
#                                freq='D')
#    
#    new = new.reindex(days)
#   and before you try it - dropna just returns the axis to monthly.
    
#    # to add the index as the legend
    ax = new.plot.line(title=title)

    if fill == True:
         minr = new.min(axis=1, numeric_only=True)
         maxr = new.max(axis=1, numeric_only=True)
         ax.fill_between(new.index, minr, maxr, alpha=0.1, label='min')
    
    #if year is None:
        # TODO this could be adapted to highlight the phenostages with their
        # labels
    # this is not doing anything useful for soil plot
    # xpos = [pd.to_datetime(startd), pd.to_datetime(endd)]
    # for xc in xpos:
    #     ax.axvline(x=xc, color='k', linestyle='--', label='pish')
    # ax.axvspan(xpos[0], xpos[1], facecolor='gray', alpha=0.2)

    ax.legend(labels=df[group].to_list(), loc='center left', 
              bbox_to_anchor=(1.0, 0.5))
   
    # So here I just need the end of the bare period as obviously I have the start
    
    crop_dict = soildates
    
    clrs = cm.rainbow(np.linspace(0, 1, len(soildates)))
    
    for p in soildates:
         plt.axvline(p, color='k', linestyle='--')
    # The crop dicts -approx patterns of crop growth....
    # this is a hacky mess for now
    # Problem is axis is monthly from previous
    # so changing to days or weeks doesn't work, including half months
    # crop_dict = {"SP BA": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
    #                        pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
    #                        pd.to_datetime(yr+"-09-01")]}
    
    
    # crop_dict = {"SP BA": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
    #                        pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
    #                        pd.to_datetime(yr+"-09-01")],
    #              "SP BE": [],
    #              "SP Oats": [],
    #              "SP WH": [pd.to_datetime(yr+"-04-01"), pd.to_datetime(yr+"-05-01"),
    #                        pd.to_datetime(yr+"-06-01"), pd.to_datetime(yr+"-08-01"),
    #                        pd.to_datetime(yr+"-09-01")],
    #              "W Rye": [],
    #              "W Spelt": [],
    #              "W WH": []}
    
    # these are placeholders and NOT correct!!!
    # mind the list much be same length otherwise zip will stop at end of shortest
    # also the last entry is of course not between lines
    # crop_seq = {"SP BA": [' Em', ' Stem' ,' Infl',
    #                        ' Flr>\n Gr>\n Sen',
    #                        ""],
    #              "SP BE": [],
    #              "SP Oats": [],
    #              "SP WH": [' Em', ' Stem' ,' Infl',
    #                        ' Flr>\n Gr> \nSen',
    #                        ""],
    #              "W Rye": [],
    #              "W Spelt": [],
    #              "W WH": []}
    
    # # rainbow
    #clrs = cm.rainbow(np.linspace(0, 1, len(crop_dict[crop])))
    
    # #for the text 
    # style = dict(size=10, color='black')
    
    
    #useful stuff here
    #https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html
    
    # this is also a mess
    # Gen the min of mins for the position of text on y axis
    minr = new.min(axis=1, numeric_only=True)
    btm = minr.min() - (minr.min() / 2)
    # for xc, nm in zip(crop_dict[crop], crop_seq[crop]):
    #     ax.axvline(x=xc, color='k', linestyle='--')
    #     ax.text(xc, minr.min(), " "+nm, ha='left', **style)
        # can't get this to work as yet
#        ax.annotate(nm, xy=(xc, 0.1),
#                    xycoords='data', bbox=dict(boxstyle="round", fc="none", ec="gray"),
#                    xytext=(10, -40), textcoords='offset points', ha='center',
#                    arrowprops=dict(arrowstyle="->"), 
#                    annotation_clip=False)
   # theend = len(crop_dict[crop])-1
    
    # for idx, c in enumerate(clrs):
    #     if idx == len(clrs)-1:
    #         break
    #     ax.axvspan(crop_dict[crop][idx], crop_dict[crop][idx+1], 
    #                 facecolor=c, alpha=0.1)
        
    
    
    # so if we take the mid date point we can put the label in the middle
 
    # hmm this does not do what I had hoped, it's not in the middle....
    # midtime = xpos[0] + (xpos[1] - xpos[0])/2
    # can cheat by putting a space at the start!!!!
    #ax.text(midtime, 0.5, " crop", ha='left', **style)
    plt.show()

    
def _S1_date(entry):
    
    # TODO I'm sure there is a better way....
    # split based on the _ then take the first block of string
    dt = entry.split(sep='_')[0][0:8]
    
    # the dt entry
    oot = datetime.strptime(dt, "%Y%m%d")
    
    return oot


def _s1_tseries(geometry, start_date='2016-01-01',
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', agg='M', para=True):
    
    """
    Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    geometry: json like
             coords of point or polygon

    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    polar: string
             send receive characteristic - VV, VH or both
             
    orbit: string
             the orbit direction - either 'ASCENDING' or 'DESCENDING'
    
    agg: string
            aggregate to a pandas time signature
            eg M or W etc
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
              
    """
    # joblib hack - this is gonna need to run in gee directly i think
    if para == True:
        ee.Initialize()
        
        
    
    
    # give the point a distance to ensure we are on a pixel
    df = _get_s1_prop(start_date, end_date, geometry,
                          polar=polar, orbit=orbit, dist=dist)
    
    # a band ratio that will be useful to go here
    # the typical one that is used
    # TODO - is this really informative? - more filters required?
    if polar == 'VVVH':
        df['VVVH'] = (df['VV'] / df['VH'])
    
    # May change to this for merging below
    df = df.set_index(df['Date'])
    
    # ratio
    nd = pd.Series(df[polar])
    

    # A monthly dataframe 
    # Should be max or upper 95th looking at NDVI
    if stat == 'max':
        nd = nd.resample(rule=agg).max()
    if stat == 'perc':
        # avoid potential outliers - not relevant to SAR....
        nd = nd.resample(rule=agg).quantile(.95)
    if stat == 'mean':
        nd = nd.resample(rule=agg).mean()
    if stat == 'median':
        nd = nd.resample(rule=agg).median()
    
    # For entry to a shapefile must be this way up
    return nd.transpose()


def _get_s1_prop(start_date, end_date, geometry, polar='VVVH', orbit='both',
                 stat='mean', dist=10):
    
    """
    Get region info for a point geometry for S1 using filters
    
    Parameters
    ----------
    
    geometry: ee geometry (json like)
              
              
    polar: string
            either VV, VH or both
            
    orbit: string or list of strings
            either 'ASCENDING', 'DESCENDING' or 'both'
            
    dist: int
          distance from point in metres (e.g. for 10m pixel it'd be 10)
          
    Returns
    -------
    
    a dataframe of S1 region info:
    
    'id', 'longitude', 'latitude', 'Date', 'VV', 'VH', 'angle'
    
    """
    
    def _reduce_region(image):
        # cheers geeextract!   
        """Spatial aggregation function for a single image and a polygon feature"""
        # the scale (int) could be anything really, why 30? cos its in the middle of
        # 10-45
        stat_dict = image.reduceRegion(fun, geometry, 30);
        # FEature needs to be rebuilt because the backend doesn't accept to map
        # functions that return dictionaries
        return ee.Feature(None, stat_dict)
    
    # TODO - Needs tidied up conditional statements are a bit of a mess
    # the collection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(start_date,
                       end_date)
    
    # should we return both?
    if polar == "VVVH":
        pol_select = s1.filter(ee.Filter.eq('instrumentMode',
                                            'IW')).filterBounds(geometry)
        
        #s1f = s1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    else:
        # select only one...
        s1f = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation',
                                               polar)).filter(ee.Filter.eq(
                                                       'instrumentMode', 'IW'))
        # emit/recieve characteristics
        pol_select = s1f.select(polar).filterBounds(geometry)

    # orbit filter
    if orbit != "VVVH":
         orbf = pol_select.filter(ee.Filter.eq('orbitProperties_pass', orbit))
    else:
        orbf = pol_select
        
    if geometry['type'] == 'Polygon':
        
        # cheers geeextract folks 
        if stat == 'mean':
            fun = ee.Reducer.mean()
        elif stat == 'median':
            fun = ee.Reducer.median()
        elif stat == 'max':
            fun = ee.Reducer.max()
        elif stat == 'min':
            fun = ee.Reducer.min()
        elif stat == 'perc':
            # for now as don't think there is equiv
            fun = ee.Reducer.mean()
        else:
            raise ValueError('Must be one of mean, median, max, or min')
        geomee = ee.Geometry.Polygon(geometry['coordinates'])
        
        s1List = orbf.filterBounds(geomee).map(_reduce_region).getInfo()
        s1List = simplify(s1List)
        
        df = pd.DataFrame(s1List)
        
        df['Date'] = df['id'].apply(_conv_dateS1)
    
    elif geometry['type'] == 'Point':
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # a point
        geomee = ee.Geometry.Point(geomee)
        
        # get the point info 
        s1list = orbf.getRegion(geomee, dist).getInfo()
        
        cols = s1list[0]
        
        # stay consistent with S2 stuff
        cols[3]='Date'
        
        # the rest
        rem=s1list[1:len(s1list)]
    
        # now a big df to reduce somewhat
        df = pd.DataFrame(data=rem, columns=cols)

        # get a "proper" date - 
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')

 
    return df

def S1_ts(inshp, start_date='2016-01-01', reproj=False, 
               end_date='2016-12-31', dist=20,  polar='VVVH',
               orbit='ASCENDING', stat='mean', outfile=None, agg='M',
               para=True, nt=-1):
    
    
    """
    Sentinel 1 month time series from a point shapefile
    
    Parameters
    ----------
    
    inshp: string
            a shapefile to join the results to 
    
    reproj: bool
            whether to reproject to wgs84, lat/lon
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    polar: string
             send receive characteristic - VV, VH or VVVH 
             (if param VVVH is used the column will be called S1R, to avoid
             search confusion)
             
    orbit: string
             the orbit direction - either 'ASCENDING' or 'DESCENDING'
    
    agg: string
            aggregate to a pandas time signature
            eg M or W etc
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            
    outfile: string
               the output shapefile if required
             
    Returns
    -------
    
    geopandas dataframe
    
    Notes
    -----
    
    This spreads the point queries client side, meaning the bottleneck is the 
    of threads you have. This is maybe 'evened out' by returning the full dataframe 
    quicker than dowloading it all from the server side
    
    """
    gdf = gpd.read_file(inshp)
    
    geom = poly2dictlist(inshp, wgs84=reproj)
    
    idx = np.arange(0, len(geom))
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    #test
#    wcld = [_s1_tseries(geom[p],
#                 start_date=start_date,
#                 end_date=end_date,
#                 orbit=orbit, stat=stat, agg=agg, polar=polar,
#                 para=para) for p in tqdm(idx)]
      
    wcld = Parallel(n_jobs=nt, verbose=2)(delayed(_s1_tseries)(geom[p],
                    start_date=start_date,
                    end_date=end_date,
                    orbit=orbit, stat=stat, agg=agg, polar=polar, 
                    para=para) for p in idx) 
    
    finaldf = pd.DataFrame(wcld)
    # if agg == 'M':
    #     finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
    # else:
    finaldf.columns = finaldf.columns.strftime("%y-%m-%d").to_list()
    
    if polar == 'VVVH':
        # otherwise this screws up searches for only VH etc
        polar = 'S1R'
    
    # if agg != 'M':
    #TODO A mess to be replaced
    if polar == 'VV':
        polar = 'V'
    elif polar == 'VH':
        polar = 'H'
    elif polar == 'S1R':
        polar = 'R'
    finaldf.columns = [polar+'-'+c for c in finaldf.columns]
    # else:

    #     finaldf.columns = [polar+'-'+c for c in finaldf.columns]
        
    #newdf = pd.merge(gdf, finaldf, on=gdf.index)
    finaldf.index = gdf.index 
    newdf = pd.concat([gdf, finaldf], axis=1)
    
    if outfile != None:
        newdf.to_file(outfile)
    
    return newdf


def S2ts_eemont(inshp, start_date='2020-01-01', end_date='2021-01-01', 
                col='COPERNICUS/S2_SR',  
                bands=['NDVI'], id_field='NGFIELD', agg='M'):
    
    # limited by no of polygons / imcollection
    # error occurs in conversion to pandas....
    # either of the above can cause it eg either too many in the imcollection
    # or too many polygons
    
    gdf = gpd.read_file(inshp)
    
    if 'key_0' in gdf.columns:
        gdf = gdf.drop(columns=['key_0'])
    
    geom = geemap.gdf_to_ee(gdf)
        
    S2 = (ee.ImageCollection(col)
   .filterBounds(geom)
   .filterDate(start_date, end_date)
   #.maskClouds()
   .scaleAndOffset()
   .index(bands))
    
    # OK so with ISO date format it works, but NOT in 'ms' (miliseconds)
    ts = S2.getTimeSeriesByRegions(reducer=[ee.Reducer.mean()],
                               collection=geom,
                               bands=bands,
                               scale=10,
                               dateColumn ='date',
                               dateFormat ='ISO')
    
    
    # the limit is 5000 for geemap...(or GEE??)
    # think it is to do with printing to console...
    df = geemap.ee_to_pandas(ts)
    
    df[df==-9999]=np.nan
    
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    # works ok - lots of missing values of course could be filled/smoothed. 
#    df = df.set_index('date')
#    df.NDVI.plot.line()
    
    # Seems to work, but adding the corresponding dates is a pain...
    yip = df.groupby(id_field)['NDVI'].apply(list).apply(pd.Series)
    # likely a silly way to do this....
    dates = df.groupby(id_field)['date'].apply(list).apply(pd.Series)
    dtrow  = dates.iloc[0].tolist()
    del dates
    yip.columns = dtrow
    
    newdf = pd.merge(gdf, yip, on=gdf.index)
    
    # if outfile != None:
    #     newdf.to_file(outfile)
    
    return newdf
    
    # if a monthly agg
    #yip.columns = yip.columns.strftime("%y-%m").to_list() 
    #yip.columns = ["nd-"+c for c in yip.columns]
    
    

    

def gdf2ee(gdf):
    
    
    features = []
    
    for i in range(gdf.shape[0]):
        geom = gdf.iloc[i:i+1,:] 
        jsonDict = geom.to_json()
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict)) 
    return features

    
# make a list of features
        
def _point2geefeat(points):
    
    """
    convert points to GEE feature collection
    """
    
    # there must be a better way - quick enough though
    feats = [ee.Feature(ee.Geometry.Point(p), {'id': str(idx)} ) for idx, p in enumerate(points)] 
      
    fcoll = ee.FeatureCollection(feats)
    
    return fcoll


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



# a notable alternative
#import ee
#ee.Initialize()
#

#using fcoll above
    
#points = list(zip(lons, lats))
#
#fcoll = _point2geefeat(points)
#
#collection = ee.ImageCollection(
#    'MODIS/006/MOD13Q1').filterDate('2017-01-01', '2017-05-01')
#
#def setProperty(image):
#    dict = image.reduceRegion(ee.Reducer.mean(), fcoll)
#    return image.set(dict)
#
#withMean = collection.map(setProperty)
#
#yip = withMean.aggregate_array('NDVI').getInfo()

#def zonal_tseries(inShp, collection, start_date, end_date, outfile,  
#                  bandnm='NDVI', attribute='id'):
#    
#    """
#    Zonal Time series for a feature collection 
#    
#    Parameters
#    ----------
#    
#    inShp: string
#                a shapefile in the WGS84 lat/lon proj
#              
#    collection: string
#                    the image collection  best if this is agg'd monthly or 
#                    something
#    
#    start_date: string
#                    start date of time series
#    
#    end_date: string
#                    end date of time series
#             
#    bandnm: string
#             the bandname of choice that exists or has been created in 
#             the image collection  e.g. B1 or NDVI
#            
#    attribute: string
#                the attribute for filtering (required for GEE)
#                
#    Returns
#    -------
#    
#    shapefile (saved) and pandas dataframe
#    
#    Notes
#    -----
#    
#    Unlike the other tseries functions here, this operates server side, meaning
#    the bottleneck is in the download/conversion to shapefile/geojson
#    """
#    
#    
#    shp = geemap.shp_to_ee(inShp)
#    
#    # name the img
#    def rename_band(img):
#        return img.select([0], [img.id()])
#    
#    
#    stacked_image = collection.map(rename_band).toBands()
#    
#    # get the img scale
#    scale = collection.first().projection().nominalScale()
#    
#    # the finished feat collection
#    ts = ee.Image(stacked_image).reduceRegions(collection=shp,
#                 reducer=ee.Reducer.mean(), scale=scale)
#    
#    geemap.ee_export_vector(ts, outfile)
#    
#    # TODO return a dataframe?
#    gdf = gpd.read_file(outfile)
#    
#    return gdf 
