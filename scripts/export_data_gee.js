// Google Earth Engine script for data export
/**
 * @name Export_Sentinel2_Wildfire_Data
 * @description Exports 10m Sentinel-2 imagery for Wildfire MVP.
 * Focus: Alaska Fire Season 2021.
 */

// 1. Define Area of Interest (ROI) - Example: Use a point in Alaska and buffer it
// You can also draw a rectangle in the map and rename it 'geometry'
var roi = ee.Geometry.Point([-150.0, 64.0]).buffer(5000).bounds();

// 2. Load Sentinel-2 Surface Reflectance
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");

// 3. Function to mask clouds
function maskS2clouds(image) {
  var qa = image.select('QA60');
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// 4. Filter Collection (Summer 2021 - Fire Season)
var dataset = s2.filterDate('2021-06-01', '2021-08-30')
                .filterBounds(roi)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(maskS2clouds);

// 5. Select 10m Bands (B2=Blue, B3=Green, B4=Red, B8=NIR)
var image = dataset.median().select(['B4', 'B3', 'B2']);

// 6. Visualization (Optional - checks if data looks good on map)
Map.centerObject(roi, 12);
Map.addLayer(image, {min: 0, max: 0.3, bands: ['B4', 'B3', 'B2']}, 'RGB');

// 7. Export to Drive
Export.image.toDrive({
  image: image,
  description: 's2_2021_06_input_10m',
  scale: 10,  // Critical: Keeps it at 10m resolution
  region: roi,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

print("Export task created! Check the 'Tasks' tab.");