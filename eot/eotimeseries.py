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
import ee, eemont
from eot.s2_masks import addCloudShadowMask, applyCloudShadowMask, addGEOS3Mask
from eot.s2_fcover import fcover
ee.Initialize()

ogr.UseExceptions()
osr.UseExceptions()

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
    
    S2 collection with s2 cloudless and fractional cover included
    
    """
    
    def _funcbs(img):
        
        geos3 = addGEOS3Mask(img)
        return img.addBands(geos3)
        # originally this updated the mask meaning only the bare pixels were returned for all bands
        # return img.updateMask(geos3)
        return img.addBands(geos3)

    bs_collection = collection.map(_funcbs)
    return bs_collection


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
    
    GEE collection of monthly NDVI images
    
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
    
    return ext


def extent2poly(infile, filetype='polygon', outfile=True, polytype="ESRI Shapefile", 
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
        vds = ogr.Open(infile)
        lyr = vds.GetLayer()
        ext = lyr.GetExtent()
    
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
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
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
    if outfile != None:
        outfile = infile[:-4]+'extent.shp'
        
        out_drv = ogr.GetDriverByName(polytype)
        
        # remove output shapefile if it already exists
        if os.path.exists(outfile):
            out_drv.DeleteDataSource(outfile)
        
        # create the output shapefile
        ootds = out_drv.CreateDataSource(outfile)
        ootlyr = ootds.CreateLayer("extent", tproj, geom_type=ogr.wkbPolygon)
        
        # add an ID field
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        ootlyr.CreateField(idField)
        
        # create the feature and set values
        featureDefn = ootlyr.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", 1)
        ootlyr.CreateFeature(feature)
        feature = None
        
        # Save and close 
        ootds.FlushCache()
        ootds = None
    
        geemap.shp_to_ee(outfile)
    
    # flatten to 2d (done in place)
    poly.FlattenTo2D()
    
    # mind it is a string before this!!!
    ootds = json.loads(poly.ExportToJson())

    return ootds



def zonal_tseries(collection, start_date, end_date, inShp, bandnm='NDVI',
                  attribute='id', scale=20):
    
    
    """
    Zonal Time series for a feature collection 
    
    Parameters
    ----------
              
    collection: string
                    the image collection  best if this is agg'd monthly or 
                    something
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
             
    bandnm: string
             the bandname of choice that exists or has been created in 
             the image collection  e.g. B1 or NDVI
            
    attribute: string
                the attribute for filtering (required for GEE)
                
    Returns
    -------
    
    pandas dataframe
    
    Notes
    -----
    
    Unlike the other tseries functions here, this operates server side, meaning
    the bottleneck is in the download/conversion to dataframe.
    
    This function is not reliable with every image collection at present
    
    """
    # Overly elaborate, unreliable and fairly likely to be dumped 

    # shp/json to gee feature here
    # if #filetype is something:
    'converting to ee feature'
    shp = geemap.shp_to_ee(inShp)
    # else # it is a json:
    
    
    # select the band and perform a spatial agg
    # GEE makes things ugly/hard to read
    def _imfunc(image):
      return (image.select(bandnm)
        .reduceRegions(
          collection=shp.select([attribute]),
          reducer=ee.Reducer.mean(),
          scale=scale
        )
        .filter(ee.Filter.neq('mean', None))
        .map(lambda f: f.set('imageId', image.id())))

    # now map the above over the collection
    # we have a triplet of attributes, which can be rearranged to a table
    # below,
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
    
    # run the above to produce the table where the columns are attributes
    # and the rows the image ID
    table = _fmt(triplets, attribute, 'imageId')
    
    print('converting to pandas df')
    df = geemap.ee_to_pandas(table)
    # bin the const GEE index
    df = df.drop(columns=['system:index'])
    #Nope.....
    #geemap.ee_export_vector(table, outfile)
    
    
    return df
    

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
        
def _s2_tseries(geometry,  collection="L2A", start_date='2016-01-01',
               end_date='2016-12-31', dist=20, cloud_mask=True, cloudless=True,
               stat='max', cloud_perc=100, ndvi=True, bandlist=None, para=False,
               agg='M'):
    
    """
    S2 Time series from a single coordinate using gee
    
    Parameters
    ----------
    
    geometry: list
                either [lon,lat] fot a point, or a set of coordinates for a 
                polygon


    collection: string
                    the S2 collection either L1C (optional cld mask) 
                    or L2A (cld masked by default)
                    or S2Cloudless
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    cloud_mask: int
             whether to mask cloud

    cloudless: bool
             whether to use S2 cloudless to mask clouds
             
    cloud_perc: int
             the acceptable cloudiness per pixel in addition to prev arg

    agg: string
            aggregate to a pandas time signature
            eg M or W etc
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliase
              
    """
    # joblib hack - this is goona need to run in gee directly i think
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
    
    if collection == 'L1C':
        col ="COPERNICUS/S2" 
        S2 = ee.ImageCollection(col).filterDate(start_date,
                           end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                           cloud_perc))
    elif collection == 'L2A':
        col ="COPERNICUS/S2_SR"                                                     
        S2 = (ee.ImageCollection(col)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .maskClouds()
        #.scaleAndOffset()
        .spectralIndices(['NDVI']))
        
    if cloudless == True:
        S2 = s2cloudless(S2, start_date, end_date, geometry,
                        cloud_filter=60)
                       
    
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
        
        s2List = S2.filterBounds(geomee).map(_reduce_region).getInfo()
        s2List = simplify(s2List)
        
        df = pd.DataFrame(s2List)
        
    elif geometry['type'] == 'Point':
    
        # a point
        geomee = ee.Geometry.Point(geometry['coordinates'])
        # give the point a distance to ensure we are on a pixel
        s2list = S2.filterBounds(geomee).getRegion(geomee, dist).getInfo()
    
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
    
    if cloud_mask == True and collection == 'L1C':
        
        # Have kept the bitwise references here
        cloudBitMask = 1024 #1 << 10 
        cirrusBitMask = 2048 # 1 << 11 
        
        df = df.drop(df[df.QA60 == cloudBitMask].index)
        df = df.drop(df[df.QA60 == cirrusBitMask].index)
        # could be done on server....
        df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    
    # May change to this for merging below
    df = df.set_index(df['Date'])
    # due to the occasional bug being an object
    if ndvi == True and bandlist == None:
        nd = pd.to_numeric(pd.Series(df['NDVI']))
    elif ndvi == True and bandlist != None:
        #TODO why have done the above when it is simpler to do the below
        bandlist.append('NDVI')
        nd = df[bandlist]#.to_numpy()
    elif ndvi == False and bandlist != None:
        nd = df[bandlist]
    # return nd
    # may be an idea to label the 'id' as a contant val or something
    # dump the redundant  columns
    
    # A monthly dataframe 
    # Should be max or upper 95th looking at NDVI
    
    
    if stat == 'max':
        nd = nd.resample(rule=agg).max()
    if stat == 'mean':
        nd = nd.resample(rule=agg).mean()
    if stat == 'median':
        nd = nd.resample(rule=agg).max()
    elif stat == 'perc':
        # avoid potential outliers
        nd = nd.resample(rule=agg).quantile(.95)
    
    # For entry to a shapefile must be this way up
    return nd.transpose()
    

# A quick/dirty answer but not an efficient one - this took almost 2mins....

def S2_ts(inshp, collection="L2A", reproj=False,
          start_date='2016-01-01', end_date='2016-12-31', dist=20, cloudless=True,
          cloud_mask=True,   stat='max', cloud_perc=100, ndvi=True,#
          bandlist=None, para=False, outfile=None, nt=-1,
               agg='M'):
    
    
    """
    Monthly time series from a point shapefile 
    
    Parameters
    ----------
             
    inshp: string
                a shapefile to join the results to 
              
    collection: string
                    the S2 collection either L1C (optional cld mask) 
                    or L2A (cld masked by default)
    
    reproj: bool
                whether to reproject to wgs84, lat/lon
    
    start_date: string
                    start date of time series
    
    end_date: string
                    end date of time series
                    
    dist: int
             the distance around point e.g. 20m
             
    cloud_mask: int
             whether to mask cloud
             
    cloud_perc: int
             the acceptable cloudiness per pixel in addition to prev arg
    
    outfile: string
           the output shapefile if required

    agg: string
            aggregate to a pandas time signature
            eg M or W etc
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliase
             
             
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
                    collection=collection,
                    start_date=start_date,
                    end_date=end_date,
                    stat=stat,
                    cloud_perc=cloud_perc,
                    cloud_mask=cloud_mask,
                    cloudless=cloudless,
                    ndvi=ndvi, 
                    bandlist=bandlist,
                    para=True,
                    agg=agg) for p in idx) 
    
    if bandlist != None:
        
        #TODO There must be a more elegant/efficient way
        listarrs = [d.to_numpy().flatten() for d in datalist]
        finaldf = pd.DataFrame(np.vstack(listarrs))
        del listarrs
        
        colstmp = []
        # this seems inefficient
        times = datalist[0].columns.strftime("%y-%m").tolist() #* (len(bandlist)+1)
        for b in bandlist:
            tmp = [b+"-"+ t for t in times]
            colstmp.append(tmp)
        # apperently quickest
        colsfin = list(chain.from_iterable(colstmp))
        
        finaldf.columns = colsfin

        
    else:
    
        finaldf = pd.DataFrame(datalist)
        
        if agg == 'M':
            finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
        else:
            finaldf.columns = finaldf.columns.strftime("%y-%m-%d").to_list()
    

        finaldf.columns = ["n-"+c for c in finaldf.columns]

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
    sqr = df.loc[df[group].isin(index)]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq=freq)
    else:
        
        # set the dtrange......
        # this code is dire  due to way i wrote other stuff from gee plus
        # ad hoc changes as I go
        #TODO alter this crap
        if freq == 'M':
            startd = yrcols[0][-5:]
            startd = '20'+startd+'-01'
            # this seems stupid
            endd = yrcols[-1:][0][-5:]
            endd = '20'+endd[0:2]+'-12-31'
            dtrange = pd.date_range(start=startd, end=endd, freq=freq)
            
        else:
            startd = yrcols[0][-8:]
            startd = '20'+startd
            endd = yrcols[-1:][0][-5:]
            endd = '20'+endd


            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    
    new = ndplotvals.transpose()
    
    if freq != 'M':
        new['Date'] = new.index
        new['Date'] = new['Date'].str.replace(name,'20')
        new['Date'] = new['Date'].str.replace('_','0')
        new['Date'] = pd.to_datetime(new['Date'])
    else:
        new['Date'] = dtrange
    
    
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
    

    sqr = df.loc[df[group].isin(index)]
    
    yrcols = [y for y in sqr.columns if name in y]
    
    if year != None:
        if len(year) > 2:
            raise ValueError('year must be last 2 digits eg 16 for 2016') 
        yrcols = [y for y in yrcols if year in y]
        dtrange = pd.date_range(start='20'+year+'-01-01',
                                end='20'+year+'-12-31',
                                freq='M')
    else:
        # set the dtrange......
        # this code is dire but due to way i wrote other stuff from gee
        
        if freq == 'M':
            startd = yrcols[0][-5:]
            startd = '20'+startd+'-01'
            # this seems stupid
            endd = yrcols[-1:][0][-8:]
            endd = '20'+endd[0:2]+'-12-31'
            dtrange = pd.date_range(start=startd, end=endd, freq=freq)

            
    # TODO - this is crap really needs replaced....
    ndplotvals = sqr[yrcols]
    
    new = ndplotvals.transpose()
    
    if freq != 'M':
        new['Date'] = new.index
        new['Date'] = new['Date'].str.replace(name+'-','20')
        new['Date'] = new['Date'].str.replace('_','0')
        new['Date'] = pd.to_datetime(new['Date'])
    else:
        new['Date'] = dtrange
    
    new = new.set_index('Date')
    new.columns=[name]
    
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
    if agg == 'M':
        finaldf.columns = finaldf.columns.strftime("%y-%m").to_list()
    else:
        finaldf.columns = finaldf.columns.strftime("%y-%m-%d").to_list()
    
    if polar == 'VVVH':
        # otherwise this screws up searches for only VH etc
        polar = 'S1R'
    
    if agg != 'M':
    #TODO A mess to be replaced
        if polar == 'VV':
            polar = 'V'
        elif polar == 'VH':
            polar = 'H'
        elif polar == 'S1R':
            polar = 'R'
        finaldf.columns = [polar+'-'+c for c in finaldf.columns]
    else:

        finaldf.columns = [polar+'-'+c for c in finaldf.columns]
        
    newdf = pd.merge(gdf, finaldf, on=gdf.index)
    
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
    
    if outfile != None:
        newdf.to_file(outfile)
    
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
