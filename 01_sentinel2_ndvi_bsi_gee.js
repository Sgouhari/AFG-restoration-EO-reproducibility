// Sentinel-2 NDVI and BSI seasonal composites
// Research code accompanying the Afghanistan restoration monitoring analysis.
//
// This script reproduces the Sentinel-2 preprocessing used for the primary
// NDVI and BSI indicators. It generates seasonal median composites for
// 2017–2024 and exports district-level GeoTIFFs plus a QA table.
//
// Before running, replace GEE_ASSET_ROOT with the Earth Engine project that
// contains the seven district boundary assets listed below.

// -----------------------------------------------------------------------------
// 1. Study areas
// -----------------------------------------------------------------------------

var GEE_ASSET_ROOT = 'projects/YOUR_GEE_PROJECT/assets/';

var districtConfigs = [
  {asset: GEE_ASSET_ROOT + 'Bamyan_district_Bamyan', district: 'Bamyan', fileDistrict: 'Bamyan', province: 'Bamyan'},
  {asset: GEE_ASSET_ROOT + 'Sayghan_district_Bamyan', district: 'Saighan', fileDistrict: 'Saighan', province: 'Bamyan'},
  {asset: GEE_ASSET_ROOT + 'Miramor_district_Daykundi', district: 'Miramor', fileDistrict: 'Miramor', province: 'Daikundi'},
  {asset: GEE_ASSET_ROOT + 'Shahristan_district_Daykundi', district: 'Shahristan', fileDistrict: 'Shahristan', province: 'Daikundi'},
  {asset: GEE_ASSET_ROOT + 'HazarSumuch_district_Takhar', district: 'Hazar Sumuch', fileDistrict: 'HazarSumuch', province: 'Takhar'},
  {asset: GEE_ASSET_ROOT + 'Kalfagan_district_Takhar', district: 'Kalfagan', fileDistrict: 'Kalfagan', province: 'Takhar'},
  {asset: GEE_ASSET_ROOT + 'Rustaq_district_Takhar', district: 'Rustaq', fileDistrict: 'Rustaq', province: 'Takhar'}
];

var districtFeatures = districtConfigs.map(function(cfg) {
  var boundary = ee.FeatureCollection(cfg.asset);
  return ee.Feature(boundary.geometry(), {
    district: cfg.district,
    file_district: cfg.fileDistrict,
    province: cfg.province
  });
});

var districts = ee.FeatureCollection(districtFeatures);

// -----------------------------------------------------------------------------
// 2. Analysis settings
// -----------------------------------------------------------------------------

var START_YEAR = 2017;
var END_YEAR = 2024;

var SEASONS = [
  {name: 'peak', startMonth: 6, endMonth: 8},
  {name: 'late', startMonth: 9, endMonth: 10}
];

var MAX_SCENE_CLOUD = 70;
var EXPORT_SCALE = 10;
var EXPORT_CRS = 'EPSG:4326';
var EXPORT_FOLDER = 'AFG_S2_NDVI_BSI_2017_2024';

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// -----------------------------------------------------------------------------
// 3. Sentinel-2 masking and spectral indices
// -----------------------------------------------------------------------------

function maskSentinel2(image) {
  var scl = image.select('SCL');

  var clear = scl.neq(1)
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image
    .updateMask(clear)
    .copyProperties(image, image.propertyNames());
}

function addIndices(image) {
  var ndvi = image
    .normalizedDifference(['B8', 'B4'])
    .rename('NDVI');

  var bsi = image.expression(
    '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
    {
      SWIR1: image.select('B11'),
      RED: image.select('B4'),
      NIR: image.select('B8'),
      BLUE: image.select('B2')
    }
  ).rename('BSI');

  return image
    .addBands(ndvi)
    .addBands(bsi)
    .addBands(ndvi.mask().rename('valid_ndvi'))
    .addBands(bsi.mask().rename('valid_bsi'))
    .copyProperties(image, image.propertyNames());
}

// -----------------------------------------------------------------------------
// 4. Seasonal composites
// -----------------------------------------------------------------------------

function buildSeasonComposite(year, season, geometry) {
  var startDate = ee.Date.fromYMD(year, season.startMonth, 1);
  var endDate = ee.Date.fromYMD(year, season.endMonth, 1).advance(1, 'month');

  var collection = S2
    .filterBounds(geometry)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', MAX_SCENE_CLOUD))
    .map(maskSentinel2)
    .map(addIndices);

  var ndvi = collection.select('NDVI').median().rename('NDVI');
  var bsi = collection.select('BSI').median().rename('BSI');
  var ndviValidCount = collection.select('valid_ndvi').sum().rename('NDVI_valid_count');
  var bsiValidCount = collection.select('valid_bsi').sum().rename('BSI_valid_count');

  return ndvi
    .addBands(bsi)
    .addBands(ndviValidCount)
    .addBands(bsiValidCount)
    .clip(geometry)
    .set({
      year: year,
      season: season.name,
      start_month: season.startMonth,
      end_month: season.endMonth,
      image_count: collection.size()
    });
}

// -----------------------------------------------------------------------------
// 5. QA summary
// -----------------------------------------------------------------------------

function buildQaFeature(districtFeature, composite) {
  var geometry = districtFeature.geometry();

  var indexStats = composite.select(['NDVI', 'BSI']).reduceRegion({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
    geometry: geometry,
    scale: EXPORT_SCALE,
    maxPixels: 1e13,
    bestEffort: true
  });

  var countStats = composite
    .select(['NDVI_valid_count', 'BSI_valid_count'])
    .reduceRegion({
      reducer: ee.Reducer.mean()
        .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
        .combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
      geometry: geometry,
      scale: EXPORT_SCALE,
      maxPixels: 1e13,
      bestEffort: true
    });

  return ee.Feature(null, {
    province: districtFeature.get('province'),
    district: districtFeature.get('district'),
    year: composite.get('year'),
    season: composite.get('season'),
    start_month: composite.get('start_month'),
    end_month: composite.get('end_month'),
    image_count: composite.get('image_count'),

    ndvi_mean: indexStats.get('NDVI_mean'),
    ndvi_min: indexStats.get('NDVI_min'),
    ndvi_max: indexStats.get('NDVI_max'),
    ndvi_stddev: indexStats.get('NDVI_stdDev'),

    bsi_mean: indexStats.get('BSI_mean'),
    bsi_min: indexStats.get('BSI_min'),
    bsi_max: indexStats.get('BSI_max'),
    bsi_stddev: indexStats.get('BSI_stdDev'),

    ndvi_valid_mean: countStats.get('NDVI_valid_count_mean'),
    ndvi_valid_min: countStats.get('NDVI_valid_count_min'),
    ndvi_valid_max: countStats.get('NDVI_valid_count_max'),

    bsi_valid_mean: countStats.get('BSI_valid_count_mean'),
    bsi_valid_min: countStats.get('BSI_valid_count_min'),
    bsi_valid_max: countStats.get('BSI_valid_count_max')
  });
}

// -----------------------------------------------------------------------------
// 6. Export seasonal NDVI and BSI composites
// -----------------------------------------------------------------------------

var districtList = districts.toList(districts.size());
var districtCount = districts.size().getInfo();
var qaFeatures = [];

for (var i = 0; i < districtCount; i++) {
  var districtFeature = ee.Feature(districtList.get(i));
  var geometry = districtFeature.geometry();

  var province = districtFeature.get('province').getInfo();
  var district = districtFeature.get('file_district').getInfo();

  for (var year = START_YEAR; year <= END_YEAR; year++) {
    for (var s = 0; s < SEASONS.length; s++) {
      var season = SEASONS[s];
      var composite = buildSeasonComposite(year, season, geometry);
      var prefix = province + '_' + district;

      qaFeatures.push(buildQaFeature(districtFeature, composite));

      Export.image.toDrive({
        image: composite.select('NDVI'),
        description: prefix + '_NDVI_' + season.name + '_' + year,
        folder: EXPORT_FOLDER,
        fileNamePrefix: prefix + '_NDVI_' + season.name + '_' + year,
        region: geometry,
        scale: EXPORT_SCALE,
        crs: EXPORT_CRS,
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });

      Export.image.toDrive({
        image: composite.select('BSI'),
        description: prefix + '_BSI_' + season.name + '_' + year,
        folder: EXPORT_FOLDER,
        fileNamePrefix: prefix + '_BSI_' + season.name + '_' + year,
        region: geometry,
        scale: EXPORT_SCALE,
        crs: EXPORT_CRS,
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });
    }
  }
}

// -----------------------------------------------------------------------------
// 7. Export QA table
// -----------------------------------------------------------------------------

var qaTable = ee.FeatureCollection(qaFeatures);

Export.table.toDrive({
  collection: qaTable,
  description: 'AFG_S2_NDVI_BSI_QA_2017_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'AFG_S2_NDVI_BSI_QA_2017_2024',
  fileFormat: 'CSV'
});

print('Study districts', districts);
print('QA preview', qaTable.limit(10));
