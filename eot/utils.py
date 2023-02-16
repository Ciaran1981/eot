#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:40:23 2021

@author: Ciaran Robb

Some gdal utils borrowed from my lib and misc funcs
"""
from osgeo import gdal, ogr, osr
from tqdm import tqdm
import os 
import numpy as np
import hvplot.xarray
import holoviews as hv
import xarray as xr
from holoviews import opts
from holoviews import stream


def ndvi_roi_plot(xrds):
    
    """
    Plot an interactive time series of NDVI using holoviews where the left pane is
    the time(z) series cube of imagery and the right a plot of NDVI curves
    as selected by the user via roi. Intended for use in Jupyter.
    
    Parameters
    ----------
    
    xrds: xarray dataset
          inut xarray image stack - must have dates etc. 
    """
    
    #figure opts
    opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='viridis', width=400, height=400),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=600),
    opts.Overlay(show_legend=False))
    
    ds = hv.Dataset(dsfl)
    
    polys = hv.Polygons([])
    box_stream = streams.BoxEdit(source=polys)

    def roi_curves(data):
        if not data or not any(len(d) for d in data.values()):
            return hv.NdOverlay({0: hv.Curve([], 'time', 'NDVI')})
        
        curves = {}
        data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
        for i, (x0, x1, y0, y1) in enumerate(data):
            selection = ds.select(x=(x0, x1), y=(y0, y1))
            curves[i] = hv.Curve(selection.aggregate('time', np.mean))
        return hv.NdOverlay(curves)
    
    hlines = hv.HoloMap({i: hv.VLine(i) for i in range(9)}, 'time') # gives vertical line on plot
    dmap = hv.DynamicMap(roi_curves, streams=[box_stream])
    
    
 
def _fieldexist(vlyr, field):
    """
    check a field exists
    """
    
    lyrdef = vlyr.GetLayerDefn()

    fieldz = []
    for i in range(lyrdef.GetFieldCount()):
        fieldz.append(lyrdef.GetFieldDefn(i).GetName())
    return field in fieldz

def batch_translate_adf(inlist):
    
    """
    batch translate a load of gdal files from some format to tif
    
    Parameters
    ----------
    
    inlist: string
        A list of raster paths
    
    Returns
    -------
    
    List of file paths
    
    """
    outpths = []
    
    for i in tqdm(inlist):
        hd, _ = os.path.split(i)
        ootpth = hd+".tif"
        srcds = gdal.Open(i)
        out = gdal.Translate(ootpth, srcds)
        out.FlushCache()
        out = None
        outpths.append(ootpth)
    return outpths

def zonal_point(inShp, inRas, bandname, band=1, nodata_value=0, write_stat=True):
    
    """ 
    Get the pixel val at a given point and write to vector
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster

    band: int
           an integer val eg - 2
                            
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    
    
   

    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
    vlyr = vds.GetLayer(0)
    
    if write_stat != None:
        # if the field exists leave it as ogr is a pain with dropping it
        # plus can break the file
        if _fieldexist(vlyr, bandname) == False:
            vlyr.CreateField(ogr.FieldDefn(bandname, ogr.OFTReal))
    
    
    
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    for label in tqdm(features):
    
            if feat is None:
                continue
            
            # the vector geom
            geom = feat.geometry()
            
            #coord in map units
            mx, my = geom.GetX(), geom.GetY()  

            # Convert from map to pixel coordinates.
            # No rotation but for this that should not matter
            px = int((mx - rgt[0]) / rgt[1])
            py = int((my - rgt[3]) / rgt[5])
            
            
            src_array = rb.ReadAsArray(px, py, 1, 1)

            if src_array is None:
                # unlikely but if none will have no data in the attribute table
                continue
            outval =  int(src_array.max())
            
#            if write_stat != None:
            feat.SetField(bandname, outval)
            vlyr.SetFeature(feat)
            feat = vlyr.GetNextFeature()
        
    if write_stat != None:
        vlyr.SyncToDisk()



    vds = None
    rds = None

def batch_gdaldem(inlist, prop='aspect'):
    
    """
    batch dem calculation a load of gdal files from some format to tif
    
    Parameters
    ----------
    
    inlist: string
        A list of raster paths
    
    prop: string
        one of "hillshade", "slope", "aspect", "color-relief", "TRI",
        "TPI", "Roughness"
    
    Returns
    -------
    
    List of file paths
    
    """
    
    outpths = []
    
    for i in tqdm(inlist):
        
        ootpth = i[:-4]+prop+".tif"
        srcds = gdal.Open(i)
        out = gdal.DEMProcessing(ootpth, srcds, prop)
        out.FlushCache()
        out = None
        outpths.append(ootpth)
    return outpths

        
def replace_str(template, t):
    """
    replace strings for nextmap downloads
    """
    out1 = template.replace('hp', t[0:2])
    out2 = out1.replace('40', t[2:4])
    return out2

def clip_raster(inRas, inShp, outRas, cutline=True):

    """
    Clip a raster
    
    Parameters
    ----------
        
    inRas: string
            the input image 
            
    outPoly: string
              the input polygon file path 
        
    outRas: string (optional)
             the clipped raster
             
    cutline: bool (optional)
             retain raster values only inside the polygon       
            
   
    """
    

    vds = ogr.Open(inShp)
           
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    
    lyr = vds.GetLayer()

    
    extent = lyr.GetExtent()
    
    extent = [extent[0], extent[2], extent[1], extent[3]]
            

    print('cropping')
    ootds = gdal.Warp(outRas,
              rds,
              format = 'GTiff', outputBounds = extent)
              
        
    ootds.FlushCache()
    ootds = None
    rds = None
    
    if cutline == True:
        
        rds1 = gdal.Open(outRas, gdal.GA_Update)
        rasterize(inShp, outRas, outRas[:-4]+'mask.tif', field=None,
                  fmt="Gtiff")
        
        mskds = gdal.Open(outRas[:-4]+'mask.tif')
        
        mskbnd = mskds.GetRasterBand(1)

        cols = mskds.RasterXSize
        rows = mskds.RasterYSize

        blocksizeX = 256
        blocksizeY = 256
        
        bands = rds1.RasterCount
        
        mskbnd = mskds.GetRasterBand(1)
        
        for i in tqdm(range(0, rows, blocksizeY)):
                if i + blocksizeY < rows:
                    numRows = blocksizeY
                else:
                    numRows = rows -i
            
                for j in range(0, cols, blocksizeX):
                    if j + blocksizeX < cols:
                        numCols = blocksizeX
                    else:
                        numCols = cols - j
                    for band in range(1, bands+1):
                        
                        bnd = rds1.GetRasterBand(band)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        mask = mskbnd.ReadAsArray(j, i, numCols, numRows)
                        
                        array[mask!=1]=0
                        bnd.WriteArray(array, j, i)
                        
        rds1.FlushCache()
        rds1 = None