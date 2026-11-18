// CHIRPS monthly precipitation by project district
// Research code accompanying the Afghanistan restoration monitoring analysis.
//
// Monthly precipitation totals are derived from CHIRPS Daily and summarised
// across the seven FREESIA project districts. The full 1981-2025 record is
// retained because it is the fitting record used by the subsequent SPI
// calculation. Manuscript summaries use data through 2024.
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
    districtKey: 'Bamyan_district_Bamyan',
    district: 'Bamyan',
    province: 'Bamyan',
    order: 1
  },
  {
    asset: GEE_ASSET_ROOT + 'Sayghan_district_Bamyan',
    districtKey: 'Sayghan_district_Bamyan',
    district: 'Saighan',
    province: 'Bamyan',
    order: 2
  },
  {
    asset: GEE_ASSET_ROOT + 'Miramor_district_Daykundi',
    districtKey: 'Miramor_district_Daykundi',
    district: 'Miramor',
    province: 'Daikundi',
    order: 3
  },
  {
    asset: GEE_ASSET_ROOT + 'Shahristan_district_Daykundi',
    districtKey: 'Shahristan_district_Daykundi',
    district: 'Shahristan',
    province: 'Daikundi',
    order: 4
  },
  {
    asset: GEE_ASSET_ROOT + 'HazarSumuch_district_Takhar',
    districtKey: 'HazarSumuch_district_Takhar',
    district: 'Hazar Sumuch',
    province: 'Takhar',
    order: 5
  },
  {
    asset: GEE_ASSET_ROOT + 'Kalfagan_district_Takhar',
    districtKey: 'Kalfagan_district_Takhar',
    district: 'Kalfagan',
    province: 'Takhar',
    order: 6
  },
  {
    asset: GEE_ASSET_ROOT + 'Rustaq_district_Takhar',
    districtKey: 'Rustaq_district_Takhar',
    district: 'Rustaq',
    province: 'Takhar',
    order: 7
  }
];


// -----------------------------------------------------------------------------
// 2. Analysis settings
// -----------------------------------------------------------------------------

var START_DATE = ee.Date('1981-01-01');
var END_DATE = ee.Date('2026-01-01');  // exclusive; includes all of 2025

var TARGET_CRS = 'EPSG:32642';
var REDUCE_SCALE_M = 5500;

var EXPORT_FOLDER = 'AFG_CHIRPS_MONTHLY';
var EXPORT_NAME = 'AFG_CHIRPS_monthly_7districts_1981_2025';

var CHIRPS_DAILY = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(START_DATE, END_DATE);


// -----------------------------------------------------------------------------
// 3. Calendar-season label
// -----------------------------------------------------------------------------

function monthToSeason(month) {
  month = ee.Number(month);

  return ee.Algorithms.If(
    month.eq(12).or(month.lte(2)),
    'DJF',
    ee.Algorithms.If(
      month.lte(5),
      'MAM',
      ee.Algorithms.If(
        month.lte(8),
        'JJA',
        'SON'
      )
    )
  );
}


// -----------------------------------------------------------------------------
// 4. Build monthly CHIRPS precipitation totals
// -----------------------------------------------------------------------------

function buildMonthlyChirps(startDate, endDate) {
  var nMonths = endDate
    .difference(startDate, 'month')
    .toInt();

  var monthStarts = ee.List
    .sequence(0, nMonths.subtract(1))
    .map(function(index) {
      return startDate.advance(
        ee.Number(index),
        'month'
      );
    });

  return ee.ImageCollection(
    monthStarts.map(function(dateValue) {
      var start = ee.Date(dateValue);
      var end = start.advance(1, 'month');

      var year = start.get('year');
      var month = start.get('month');

      var monthlyTotal = CHIRPS_DAILY
        .filterDate(start, end)
        .sum()
        .rename('rain_mm')
        .set({
          year: year,
          month: month,
          ym_key: ee.Number(year)
            .multiply(100)
            .add(ee.Number(month)),
          daily_image_count: CHIRPS_DAILY
            .filterDate(start, end)
            .size(),
          'system:time_start': start.millis()
        });

      return monthlyTotal;
    })
  );
}

var CHIRPS_MONTHLY = buildMonthlyChirps(
  START_DATE,
  END_DATE
);


// -----------------------------------------------------------------------------
// 5. District-level monthly precipitation
// -----------------------------------------------------------------------------

function monthlyTableForDistrict(config) {
  var boundary = ee.FeatureCollection(config.asset);
  var geometry = boundary.geometry();

  var areaKm2 = geometry
    .area(1)
    .divide(1e6);

  var monthlyFeatures = CHIRPS_MONTHLY.map(function(image) {
    var year = ee.Number(image.get('year'));
    var month = ee.Number(image.get('month'));

    var stats = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geometry,
      scale: REDUCE_SCALE_M,
      crs: TARGET_CRS,
      maxPixels: 1e13,
      bestEffort: true
    });

    var ymKey = ee.Number(image.get('ym_key'));

    return ee.Feature(null, {
      district_key: config.districtKey,
      province: config.province,
      district: config.district,
      year: year,
      month: month,
      season: monthToSeason(month),
      rain_mm: stats.get('rain_mm'),
      aoi_km2: areaKm2,
      daily_image_count: image.get('daily_image_count'),
      ym_key: ymKey,
      sort_key: ee.Number(config.order)
        .multiply(1000000)
        .add(ymKey)
    });
  });

  return ee.FeatureCollection(monthlyFeatures);
}


// -----------------------------------------------------------------------------
// 6. Combine the seven district tables
// -----------------------------------------------------------------------------

var allMonthly = ee.FeatureCollection([]);

districtConfigs.forEach(function(config) {
  allMonthly = allMonthly.merge(
    monthlyTableForDistrict(config)
  );
});

allMonthly = allMonthly.sort('sort_key');


// -----------------------------------------------------------------------------
// 7. Export one combined monthly table
// -----------------------------------------------------------------------------

Export.table.toDrive({
  collection: allMonthly,
  description: EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: [
    'district_key',
    'province',
    'district',
    'year',
    'month',
    'season',
    'rain_mm',
    'aoi_km2',
    'daily_image_count',
    'ym_key'
  ]
});


// -----------------------------------------------------------------------------
// 8. Console checks
// -----------------------------------------------------------------------------

print('CHIRPS source', 'UCSB-CHG/CHIRPS/DAILY');
print('Monthly record', '1981-2025');
print('District count', districtConfigs.length);
print('Expected district-month rows', 7 * 45 * 12);
print('Monthly table preview', allMonthly.limit(12));
