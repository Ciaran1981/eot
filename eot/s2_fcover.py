import ee
import math

"""
Adapted from soilwatch javascript/sentinelhub script - credit to wouellette. Rewritten with python api by Ciaran Robb
FCover calculation adapted from the sentinelhub custom-scripts repository:
https:#github.com/sentinel-hub/custom-scripts/tree/a8cfd9bb1e5c9fb94aa467ea94701a2f50c0e63e/sentinel-2/fcover
Credit to Kristof van Tricht (@kristovt) for contributing the original script.
"""
def fcover(sr_band_scale=1):


  def wrap(img):

    mean_solar_az = ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
    mean_solar_zen = ee.Number(img.get('MEAN_SOLAR_ZENITH_ANGLE'))

    # Need to compute the mean mean incidence azimuth angle and mean incidence zenith angle
    # across all bands to match Sentinelhub angle values
    mean_incidence_az = ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B3')) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B4'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B5'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B6'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B7'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B11'))) \
                            .add(ee.Number(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B12'))) \
                            .divide(8)
    mean_incidence_zen = ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B3')) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B4'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B5'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B6'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B7'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B8A'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B11'))) \
                             .add(ee.Number(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B12'))) \
                             .divide(8)

    # rescale data range if necessary
    img_rs = img.divide(sr_band_scale)

    # Normalize each bands with according to the SNAP methodology
    # Input/output values which are suspect are not reported or changed, unlike in SNAP.
    b03_norm = _normalize(img_rs.select('B3'), 0, 0.253061520472)
    b04_norm = _normalize(img_rs.select('B4'), 0, 0.290393577911)
    b05_norm = _normalize(img_rs.select('B5'), 0, 0.305398915249)
    b06_norm = _normalize(img_rs.select('B6'), 0.00663797254225, 0.608900395798)
    b07_norm = _normalize(img_rs.select('B7'), 0.0139727270189, 0.753827384323)
    b8a_norm = _normalize(img_rs.select('B8A'), 0.0266901380821, 0.782011770669)
    b11_norm = _normalize(img_rs.select('B11'), 0.0163880741923, 0.493761397883)
    b12_norm = _normalize(img_rs.select('B12'), 0, 0.49302598446)
    view_zen_norm = _normalize(mean_incidence_zen.multiply(ee.Number(math.pi).divide(180)).cos(), 0.918595400582, 0.999999999991)
    sun_zen_norm  = _normalize(mean_solar_zen.multiply(ee.Number(math.pi).divide(180)).cos(), 0.342022871159, 0.936206429175)
    rel_az_norm = mean_solar_az.subtract(mean_incidence_az).multiply(ee.Number(math.pi).divide(180)).cos()

    # Apply the different layers
    n1 = _neuron1(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                      view_zen_norm, sun_zen_norm, rel_az_norm)
    n2 = _neuron2(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                      view_zen_norm, sun_zen_norm, rel_az_norm)
    n3 = _neuron3(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                      view_zen_norm, sun_zen_norm, rel_az_norm)
    n4 = _neuron4(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                      view_zen_norm, sun_zen_norm, rel_az_norm)
    n5 = _neuron5(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                      view_zen_norm, sun_zen_norm, rel_az_norm)

    # Combine the layers to get the final value.
    l2 = _layer2(n1, n2, n3, n4, n5)

    # Denormalize the data back to the original [0,1] range
    fcover = _denormalize(l2, 0.000181230723879, 0.999638214715)

    # Fetch the band type of an arbitrary spectral band to cast the fcover to the same data type
    band_type = img.select('B4').bandTypes().get('B4')

    return img.addBands(fcover.multiply(sr_band_scale).rename('fcover').cast({'fcover': band_type}))


  return wrap



#Apply the neural network for FCover
#sm formerly reserved word sum

def _neuron1(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, 
              b11_norm, b12_norm, view_zen_norm, sun_zen_norm, rel_az_norm):
    
    sm = b03_norm.multiply(-0.156854264841) \
              .add(b04_norm.multiply(0.124234528462)) \
              .add(b05_norm.multiply(0.235625516229)) \
              .subtract(b06_norm.multiply(1.8323910258)) \
              .subtract(b07_norm.multiply(0.217188969888)) \
              .add(b8a_norm.multiply(5.06933958064)) \
              .subtract(b11_norm.multiply(0.887578008155)) \
              .subtract(b12_norm.multiply(1.0808468167)) \
              .subtract(view_zen_norm.multiply(0.0323167041864)) \
              .subtract(sun_zen_norm.multiply(0.224476137359)) \
              .subtract(rel_az_norm.multiply(0.195523962947)) \
              .subtract(1.45261652206)

    return _tansig(sm)
    


def _neuron2(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                  view_zen_norm, sun_zen_norm, rel_az_norm):
    
    sm = b03_norm.multiply(-0.220824927842) \
            .add(b04_norm.multiply(1.28595395487)) \
            .add(b05_norm.multiply(0.703139486363)) \
            .subtract(b06_norm.multiply(1.34481216665)) \
            .subtract(b07_norm.multiply(1.96881267559)) \
            .subtract(b8a_norm.multiply(1.45444681639)) \
            .add(b11_norm.multiply(1.02737560043)) \
            .subtract(b12_norm.multiply(0.12494641532)) \
            .add(view_zen_norm.multiply(0.0802762437265)) \
            .subtract(sun_zen_norm.multiply(0.198705918577)) \
            .add(rel_az_norm.multiply(0.108527100527)) \
            .subtract(1.70417477557) \
    
    return _tansig(sm)



def _neuron3(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                  view_zen_norm, sun_zen_norm, rel_az_norm):
    
    sm = b03_norm.multiply(-0.409688743281) \
            .add(b04_norm.multiply(1.08858884766)) \
            .add(b05_norm.multiply(0.36284522554)) \
            .add(b06_norm.multiply(0.0369390509705)) \
            .subtract(b07_norm.multiply(0.348012590003)) \
            .subtract(b8a_norm.multiply(2.0035261881)) \
            .add(b11_norm.multiply(0.0410357601757)) \
            .add(b12_norm.multiply(1.22373853174)) \
            .subtract(view_zen_norm.multiply(0.0124082778287)) \
            .subtract(sun_zen_norm.multiply(0.282223364524)) \
            .add(rel_az_norm.multiply(0.0994993117557)) \
            .add(1.02168965849) \

    return _tansig(sm)



def _neuron4(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                  view_zen_norm, sun_zen_norm, rel_az_norm):
    
    sm = b03_norm.multiply(-0.188970957866) \
            .subtract(b04_norm.multiply(0.0358621840833)) \
            .add(b05_norm.multiply(0.00551248528107)) \
            .add(b06_norm.multiply(1.35391570802)) \
            .subtract(b07_norm.multiply(0.739689896116)) \
            .subtract(b8a_norm.multiply(2.21719530107)) \
            .add(b11_norm.multiply(0.313216124198)) \
            .add(b12_norm.multiply(1.5020168915)) \
            .add(view_zen_norm.multiply(1.21530490195)) \
            .subtract(sun_zen_norm.multiply(0.421938358618)) \
            .add(rel_az_norm.multiply(1.48852484547)) \
            .subtract(0.498002810205) \
    
    return _tansig(sm)



def _neuron5(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                  view_zen_norm, sun_zen_norm, rel_az_norm):
    
    sm = b03_norm.multiply(2.49293993709) \
            .subtract(b04_norm.multiply(4.40511331388)) \
            .subtract(b05_norm.multiply(1.91062012624)) \
            .subtract(b06_norm.multiply(0.703174115575)) \
            .subtract(b07_norm.multiply(0.215104721138)) \
            .subtract(b8a_norm.multiply(0.972151494818)) \
            .subtract(b11_norm.multiply(0.930752241278)) \
            .add(b12_norm.multiply(1.2143441876)) \
            .subtract(view_zen_norm.multiply(0.521665460192)) \
            .subtract(sun_zen_norm.multiply(0.445755955598)) \
            .add(rel_az_norm.multiply(0.344111873777)) \
            .subtract(3.88922154789) \
                
    return _tansig(sm)

# Combine the layers into the final convolution
def _layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
  sm = neuron1.multiply(0.23080586765) \
            .subtract(neuron2.multiply(0.333655484884)) \
            .subtract(neuron3.multiply(0.499418292325)) \
            .add(neuron4.multiply(0.0472484396749)) \
            .subtract(neuron5.multiply(0.0798516540739)) \
            .subtract(0.0967998147811)

  return sm
# mn was min. mx was max, inp was input - all reserved
def _normalize(unnormalized, mn, mx):
  return unnormalized.subtract(mn).multiply(2).divide(ee.Number(mx).subtract(mn)).subtract(1)

def _denormalize(normalized, mn, mx):
  return normalized.add(1).multiply(0.5).multiply(ee.Number(mx).subtract(mn)).add(mn)

def _tansig(inp):
  return ee.Image(2).divide(ee.Image(1).add(inp.multiply(-2).exp())).subtract(1)
