import ee
import geemap

ee.Authenticate()
ee.Initialize(project="")

# Create equal-area grid
def create_grid(region, scale=800000, crs='EPSG:3857', add_id=True, id_field='grid_id'):
    if isinstance(region, (ee.featurecollection.FeatureCollection, ee.feature.Feature)):
        geom = region.geometry()
    elif isinstance(region, ee.geometry.Geometry):
        geom = region
    else:
        geom = ee.Geometry(region)

    proj = ee.Projection(crs)
    grid = ee.FeatureCollection(geom.coveringGrid(proj=proj, scale=scale))

    if not add_id:
        return grid

    grid_list = grid.toList(grid.size())

    def _set_id(i):
        i = ee.Number(i)
        ft = ee.Feature(grid_list.get(i))
        return ft.set(id_field, i.add(1))

    grid_with_id = ee.FeatureCollection(ee.List.sequence(0, grid.size().subtract(1)).map(_set_id))
    return grid_with_id

roi = ee.FeatureCollection("projects/ee-2018901475/assets/ChinaBoundry_Polygon")
grid = create_grid(roi, scale=800000, crs='EPSG:3857')

Map = geemap.Map()
Map.centerObject(roi, 3)
Map.addLayer(roi, {'color': 'blue'}, 'studyArea')
Map.addLayer(grid, {}, 'Grid')
Map

# Stratified sampling by grid and class
def stratified_sample_by_grid_and_class_v2(
    sample_asset_id="projects/ee-2018901475/assets/merged_sample_points",
    grid_fc=None,
    class_field="class",
    grid_id_field="grid_id",
    n_per_grid_per_class=100,
    seed=7,
    keep_only_fields=None
):
    samples = ee.FeatureCollection(sample_asset_id)

    if keep_only_fields is not None:
        fields = ee.List(keep_only_fields).add(class_field)
        samples = samples.select(fields)

    def sample_one_grid(grid_feat):
        grid_feat = ee.Feature(grid_feat)
        gid = grid_feat.get(grid_id_field)
        geom = grid_feat.geometry()
        pts = samples.filterBounds(geom)
        local_seed = ee.Number(seed).add(ee.Number.parse(ee.String(gid)))
        pts = pts.randomColumn("rand", local_seed)

        def sample_one_class(c):
            c = ee.Number(c)
            sub = pts.filter(ee.Filter.eq(class_field, c))
            k = ee.Number(n_per_grid_per_class).min(sub.size())
            out = sub.sort("rand").limit(k)
            return out.map(lambda f: ee.Feature(f).set(grid_id_field, gid))

        return ee.FeatureCollection(ee.List([0, 1]).map(sample_one_class)).flatten()

    sampled = ee.FeatureCollection(grid_fc.map(sample_one_grid)).flatten()
    return sampled

grid_fc = create_grid(roi, scale=800000)
train_fc = stratified_sample_by_grid_and_class_v2(
    sample_asset_id="projects/ee-2018901475/assets/merged_sample_points",
    grid_fc=grid_fc,
    n_per_grid_per_class=100,
    seed=7
)
geemap.ee_export_vector_to_drive(
    train_fc,
    description="train_fc_stratified",
    folder="GEE_exports",
    fileFormat="CSV"
)

Map.addLayer(train_fc, {}, 'Stratified Sampled Points')

# Prepare CLCD image
def get_clcd_image(year, region=None):
    img_path = f"projects/lulc-datase/assets/LULC_HuangXin/CLCD_v01_{year}"
    img = ee.Image(img_path).updateMask(ee.Image(img_path).gt(0))

    if region is not None:
        if isinstance(region, ee.FeatureCollection):
            region = region.geometry()
        img = img.clip(region)

    return img

def get_clcd_impervious_mask(year, region=None):
    clcd = get_clcd_image(year, region=region)
    impervious = clcd.eq(8)
    return impervious.rename("impervious")

year = 2000
clcd = get_clcd_image(year, region=roi)
imperv = get_clcd_impervious_mask(year, region=roi)

vis_palette_clcd = [
    '#FAE39C',  # 1 cropland
    '#446F33',  # 2 forest
    '#33A02C',  # 3 shrub
    '#ABD37B',  # 4 grassland
    '#1E69B4',  # 5 water
    '#A6CEE3',  # 6 snow/ice
    '#CFBDA3',  # 7 barren
    '#E24290',  # 8 impervious
    '#289BE8',  # 9 wetland
]

Map = geemap.Map()
Map.centerObject(roi, 4)
Map.addLayer(clcd, {"min": 1, "max": 9, "palette": vis_palette_clcd}, f"CLCD {year}")
Map.addLayer(imperv.selfMask(), {"palette": ["#E24290"]}, f"Impervious {year}")
Map

# Landsat texture features
def mask_landsat8_c2_l2(image):
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return image.updateMask(mask)

def scale_landsat8_c2_l2_sr(image):
    sr = (
        image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
    )
    return image.addBands(sr, overwrite=True)

def get_landsat8_sr_composite(
    start_year, end_year, cloud_cover_max=50, composite="median", bounds=None
):
    start_date = ee.Date.fromYMD(start_year, 1, 1)
    end_date = ee.Date.fromYMD(end_year, 12, 31)

    col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte("CLOUD_COVER", cloud_cover_max)) \
        .map(mask_landsat8_c2_l2) \
        .map(scale_landsat8_c2_l2_sr) \
        .select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            ["blue", "green", "red", "nir", "swir1", "swir2"]
        )

    if bounds is not None:
        bounds = ee.Geometry(bounds)
        col = col.filterBounds(bounds)

    img = col.median() if composite == "median" else col.mean()
    return img

def add_spectral_indices(img):
    ndvi = img.normalizedDifference(["nir", "red"]).rename("NDVI")
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("NDBI")
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("MNDWI")
    bsi = img.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        {
            "blue": img.select("blue"),
            "red": img.select("red"),
            "nir": img.select("nir"),
            "swir1": img.select("swir1"),
        }
    ).rename("BSI")

    return img.addBands([ndvi, ndbi, mndwi, bsi])

def build_landsat8_features(
    start_year, end_year, cloud_cover_max=50, composite="median", use_texture=True, texture_radius_pixels=1, bounds=None
):
    img = get_landsat8_sr_composite(
        start_year=start_year,
        end_year=end_year,
        cloud_cover_max=cloud_cover_max,
        composite=composite,
        bounds=bounds
    )
    img = add_spectral_indices(img)

    if use_texture:
        texture_bands = ["NDVI", "NDBI", "nir", "swir1"]
        img = add_texture_features(
            img, bands=texture_bands, radius_pixels=texture_radius_pixels
        )

    return img

roi = ee.Geometry.Rectangle([112, 30, 114, 32])

ls_feats = build_landsat8_features(
    start_year=2016,
    end_year=2018,
    use_texture=True,
    bounds=roi
)

Map = geemap.Map()
Map.centerObject(roi, 6)
Map.addLayer(ls_feats.select("NDBI"), {"min": -0.5, "max": 0.5}, "NDBI")
Map.addLayer(ls_feats.select("NDVI_mean"), {"min": 0, "max": 0.8}, "NDVI mean (3×3)")
Map