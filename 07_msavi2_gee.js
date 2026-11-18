// Sentinel-2 MSAVI2 seasonal composites
// Research code accompanying the Afghanistan restoration monitoring analysis.
//
// This script reproduces the MSAVI2 sensitivity workflow used to assess
// potential soil-background effects. It uses the same Sentinel-2 collection,
// masking and seasonal windows as the NDVI analysis.
//
// Before running, replace GEE_ASSET_ROOT with the Earth Engine project that
// contains the seven district boundary assets listed below.

// -----------------------------------------------------------------------------
// 1. Study areas
// -----------------------------------------------------------------------------

var GEE_ASSET_ROOT = 'projects/YOUR_GEE_PROJECT/assets/';

var districtConfigs = [
  {
    asset: GEE_ASSET_ROOT + 'Bamyan_district_Bamyan',
    district: 'Bamyan',
    fileDistrict: 'Bamyan',
    province: 'Bamyan'
  },
  {
    asset: GEE_ASSET_ROOT + 'Sayghan_district_Bamyan',
    district: 'Saighan',
    fileDistrict: 'Saighan',
    province: 'Bamyan'
  },
  {
    asset: GEE_ASSET_ROOT + 'Miramor_district_Daykundi',
    district: 'Miramor',
    fileDistrict: 'Miramor',
    province: 'Daikundi'
  },
  {
    asset: GEE_ASSET_ROOT + 'Shahristan_district_Daykundi',
    district: 'Shahristan',
    fileDistrict: 'Shahristan',
    province: 'Daikundi'
  },
  {
    asset: GEE_ASSET_ROOT + 'HazarSumuch_district_Takhar',
    district: 'Hazar Sumuch',
    fileDistrict: 'HazarSumuch',
    province: 'Takhar'
  },
  {
    asset: GEE_ASSET_ROOT + 'Kalfagan_district_Takhar',
    district: 'Kalfagan',
    fileDistrict: 'Kalfagan',
    province: 'Takhar'
  },
  {
    asset: GEE_ASSET_ROOT + 'Rustaq_district_Takhar',
    district: 'Rustaq',
    fileDistrict: 'Rustaq',
    province: 'Takhar'
  }
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
var REFLECTANCE_SCALE = 0.0001;

var EXPORT_SCALE = 10;
var EXPORT_CRS = 'EPSG:4326';
var EXPORT_FOLDER = 'AFG_S2_MSAVI2_2017_2024';

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');


// -----------------------------------------------------------------------------
// 3. Sentinel-2 masking
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


// -----------------------------------------------------------------------------
// 4. MSAVI2
// -----------------------------------------------------------------------------
//
// MSAVI2 = [2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - RED))] / 2
//
// Sentinel-2 B4 (red) and B8 (NIR) are converted from stored scaled integers
// to surface reflectance before MSAVI2 is calculated. This scaling is required
// because the MSAVI2 equation contains the constant term 1.
// -----------------------------------------------------------------------------

function addMsavi2(image) {
  var red = image
    .select('B4')
    .multiply(REFLECTANCE_SCALE);

  var nir = image
    .select('B8')
    .multiply(REFLECTANCE_SCALE);

  var twoNirPlusOne = nir
    .multiply(2)
    .add(1);

  var discriminant = twoNirPlusOne
    .pow(2)
    .subtract(
      nir
        .subtract(red)
        .multiply(8)
    );

  var validMath = discriminant.gte(0);

  // Allow sqrt() to be evaluated safely, then mask pixels for which the
  // original discriminant was negative.
  var safeDiscriminant = discriminant.where(
    discriminant.lt(0),
    0
  );

  var msavi2 = twoNirPlusOne
    .subtract(safeDiscriminant.sqrt())
    .divide(2)
    .rename('MSAVI2')
    .updateMask(validMath);

  var validObservation = msavi2
    .mask()
    .rename('valid_obs');

  return image
    .addBands(msavi2)
    .addBands(validObservation)
    .copyProperties(image, image.propertyNames());
}


// -----------------------------------------------------------------------------
// 5. Seasonal composites
// -----------------------------------------------------------------------------

function buildSeasonComposite(year, season, geometry) {
  var startDate = ee.Date.fromYMD(
    year,
    season.startMonth,
    1
  );

  var endDate = ee.Date
    .fromYMD(year, season.endMonth, 1)
    .advance(1, 'month');

  var collection = S2
    .filterBounds(geometry)
    .filterDate(startDate, endDate)
    .filter(
      ee.Filter.lte(
        'CLOUDY_PIXEL_PERCENTAGE',
        MAX_SCENE_CLOUD
      )
    )
    .map(maskSentinel2)
    .map(addMsavi2);

  var msavi2 = collection
    .select('MSAVI2')
    .median()
    .rename('MSAVI2');

  var validCount = collection
    .select('valid_obs')
    .sum()
    .rename('valid_obs_count');

  return msavi2
    .addBands(validCount)
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
// 6. QA summary
// -----------------------------------------------------------------------------

function buildQaFeature(districtFeature, composite) {
  var geometry = districtFeature.geometry();

  var msavi2Stats = composite
    .select('MSAVI2')
    .reduceRegion({
      reducer: ee.Reducer.mean()
        .combine({
          reducer2: ee.Reducer.min(),
          sharedInputs: true
        })
        .combine({
          reducer2: ee.Reducer.max(),
          sharedInputs: true
        })
        .combine({
          reducer2: ee.Reducer.stdDev(),
          sharedInputs: true
        }),
      geometry: geometry,
      scale: EXPORT_SCALE,
      maxPixels: 1e13,
      bestEffort: true
    });

  var validStats = composite
    .select('valid_obs_count')
    .reduceRegion({
      reducer: ee.Reducer.mean()
        .combine({
          reducer2: ee.Reducer.min(),
          sharedInputs: true
        })
        .combine({
          reducer2: ee.Reducer.max(),
          sharedInputs: true
        }),
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

    msavi2_mean: msavi2Stats.get('MSAVI2_mean'),
    msavi2_min: msavi2Stats.get('MSAVI2_min'),
    msavi2_max: msavi2Stats.get('MSAVI2_max'),
    msavi2_stddev: msavi2Stats.get('MSAVI2_stdDev'),

    valid_obs_mean: validStats.get('valid_obs_count_mean'),
    valid_obs_min: validStats.get('valid_obs_count_min'),
    valid_obs_max: validStats.get('valid_obs_count_max')
  });
}


// -----------------------------------------------------------------------------
// 7. Export seasonal MSAVI2 composites
// -----------------------------------------------------------------------------

var districtList = districts.toList(districts.size());
var districtCount = districts.size().getInfo();
var qaFeatures = [];

for (var i = 0; i < districtCount; i++) {
  var districtFeature = ee.Feature(districtList.get(i));
  var geometry = districtFeature.geometry();

  var province = districtFeature
    .get('province')
    .getInfo();

  var district = districtFeature
    .get('file_district')
    .getInfo();

  for (var year = START_YEAR; year <= END_YEAR; year++) {
    for (var s = 0; s < SEASONS.length; s++) {
      var season = SEASONS[s];

      var composite = buildSeasonComposite(
        year,
        season,
        geometry
      );

      qaFeatures.push(
        buildQaFeature(
          districtFeature,
          composite
        )
      );

      var exportName =
        province
        + '_'
        + district
        + '_MSAVI2_'
        + season.name
        + '_'
        + year;

      Export.image.toDrive({
        image: composite.select('MSAVI2'),
        description: exportName,
        folder: EXPORT_FOLDER,
        fileNamePrefix: exportName,
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
// 8. Export QA table
// -----------------------------------------------------------------------------

var qaTable = ee.FeatureCollection(qaFeatures);

Export.table.toDrive({
  collection: qaTable,
  description: 'AFG_S2_MSAVI2_QA_2017_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'AFG_S2_MSAVI2_QA_2017_2024',
  fileFormat: 'CSV'
});

print('Study districts', districts);
print('MSAVI2 QA preview', qaTable.limit(10));
print('Reflectance scale factor', REFLECTANCE_SCALE);
