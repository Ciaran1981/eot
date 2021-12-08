#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:40:23 2021

@author: Ciaran Robb

Some gdal utils borrowed from my lib
"""
from osgeo import gdal, ogr, osr
from tqdm import tqdm
import os 
import numpy as np
 
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
