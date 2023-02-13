import ee 

"""
Adapted from the soilwatch javascript (credit wouellette). Rewritten in the python api with modifications by Ciaran Robb

"""
# Global Cloud Masking parameters
cld_prb_thresh = 40#; # Cloud probability threshold to mask clouds. 40% is the default value of s2cloudless
cloud_filter = 60#; # Threshold on sentinel-2 Metadata field determining whether cloud pixel percentage in image
nir_drk_thresh = 0.15#; # A threshold that determines when to consider a dark area a cloud shadow or not
cld_prj_dist = 10#; # The distance (in no of pixels) in which to search from detected cloud to find cloud shadows
buffer = 50#; # The cloud buffer (in meters) to use around detected cloud pixels to mask additionally
mask_res = 60#; # resolution at which to generate and apply the cloud/shadow mask. 60m instead of 10m to speed up

# Function to load Sentinel-2 data and its corresponding cloud probability information, based on an area and time range
def loadImageCollection(collection_name, time_range, geom):
  # Import Sentinel-2 L1C or L2A, depending on year selected.
  s2 = ee.ImageCollection(collection_name) \
    .filterDate(time_range.get('start'), time_range.get('end')) \
    .filterBounds(geom) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_filter)

  # Import and filter s2cloudless.
  s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
      .filterBounds(geom) \
      .filterDate(time_range.get('start'), time_range.get('end'))
  
  # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
  # bug lies here....
  # Unrecognized argument type to convert to a FeatureCollection:
  #     {'primary': <ee.imagecollection.ImageCollection object at 0x7eff2cd46610>,
  #      'secondary': <ee.imagecollection.ImageCollection object at 0x7eff2cd4f340>, 
  #      'condition': <ee.filter.Filter object at 0x7eff2cd4f940>}
  
  # To prevent the above bug the funcs must have the '**'
  s2_cl = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
      'primary': s2,
      'secondary': s2_cloudless_col,
      'condition': ee.Filter.equals(**{
          'leftField': 'system:index',
          'rightField': 'system:index'})
  })).sort('system:time_start')

  return s2_cl


# Function to combine shadow and cloud masks to image.
def addCloudShadowMask(water_valmask, sr_band_scale):
    # water_valmask : water validity mask, indicating locations of non-water pixels for the cloud shadow detection
    # sr_band_scale: scaling factor. 10000 for Sentinel-2 GEE assets

    sr_band_scale = sr_band_scale or 1 # the or replaces the jscript ||

    def wrap(img):
      # img: A sentinel-2 image

      # Add cloud component bands.
      img_cloud = _addCloudBands(img)

      # Add cloud shadow component bands.
      img_cloud_shadow = _addShadowBands(img_cloud, water_valmask, sr_band_scale)

      # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
      is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

      # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
      # 60 m scale is for speed, and assumes clouds don't require 10 m precision.
      # Morphological Opening operation is an erosion (focal_min) followed by \
      is_cld_shdw = is_cld_shdw.focal_min(2).focal_max(buffer * 2 / mask_res) \
                    .reproject(crs=img.select([0]).projection(), scale=mask_res) \
                    .rename('cloudmask')

      # Add the final cloud-shadow mask to the image.
      return img_cloud_shadow.addBands(is_cld_shdw)

    return wrap
  


 # Function to apply the final cloud mask to the image.
def applyCloudShadowMask(img):
  # img: A sentinel-2 image

  # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
  not_cld_shdw = img.select('cloudmask').Not()

  # Subset reflectance bands and update their masks, return the result.
  return img.updateMask(not_cld_shdw)



# Function to add the cloud probability band to the image based on the specified cloud probability threshold.
def _addCloudBands(img):
    # img: A sentinel-2 image

    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cld_prb_thresh).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))



# Function to add the cloud shadow mask to the image based on specified parameters.
def _addShadowBands(img, water_valmask, sr_band_scale):
    # img: A sentinel-2 image
    # water_valmask : water validity mask, indicating locations of non-water pixels for the cloud shadow detection
    # sr_band_scale: scaling factor. 10000 for Sentinel-2 GEE assets

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    dark_pixels = img.select('B8').lt(nir_drk_thresh*sr_band_scale) \
                      .multiply(water_valmask).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = img.select('clouds').directionalDistanceTransform(
        shadow_azimuth, cld_prj_dist / 10).reproject(
            crs=img.select(0).projection(),scale=mask_res).select(
                'distance').mask().rename('cloud_transform')

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))



# GEOS3 algorithm implementation from: Demattê, José AM, et al.
# "Bare earth’s Surface Spectra as a proxy for Soil Resource Monitoring." Scientific reports 10.1 (2020): 1-11.
# Applied to Sentinel-2 data.
def addGEOS3Mask(img):
    
  img_rs = img.divide(10000) # rescale to [0,1] reflectance.
  ndvi = img_rs.normalizedDifference(['B8', 'B4']) # Normalized vegetation index
  nbr2 = img_rs.normalizedDifference(['B11', 'B12']) # Normalized Burn Ratio 2
  # Visible-toshortwave-infrared tendency index
  vnsir = ee.Image(1) \
              .subtract(ee.Image(2).multiply(img_rs.select('B4')) \
                                             .subtract(img_rs.select('B3')).subtract(img_rs.select('B2')) \
              .add(ee.Image(3).multiply(img_rs.select('B12').subtract(img_rs.select('B8')))))

  # GEOS3 equation
  geos3 = ndvi.gte(-0.25).bitwiseAnd(ndvi.lte(0.25)) \
              .bitwiseAnd(nbr2.gte(-0.3).bitwiseAnd(nbr2.lte(0.1))) \
              .bitwiseAnd(vnsir.lte(0.9)).rename('GEOS3')

  return geos3 # Return bare soil pixel stack

