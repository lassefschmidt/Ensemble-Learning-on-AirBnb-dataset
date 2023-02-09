import pandas as pd
import geopandas as gpd
from shapely import wkt # to convert geodata in csvs to readable format for geopandas
import numpy as np

def show_missing_vals(df):
    (df.isna().sum()/len(df)).reset_index(name = "missing vals").plot.bar(x="index", y = "missing vals", rot = 90)

def impute_missing_vals(df, threshs = None):
    """
    Impute missing values in the provided pandas dataframe. If this function is applied to the validation / test data,
    we need to provide it all thresholds that we used to preprocess our training data. 
    """
    if threshs is None:
        threshs = {"oldest_date": min(pd.to_datetime(df.last_review)),
                    "newest_date": max(pd.to_datetime(df.last_review))}

    df = (df
        # column "host_name" (drop)
        .loc[:, df.columns != "host_name"]
        # column "last_review" (reformat & impute missing vals)
        .assign(last_review = lambda df_: pd.to_datetime(df_.last_review)) # convert "last_review" to datetime object format
        .assign(last_review = lambda df_: df_.last_review.mask(df_.last_review.isna(), threshs["oldest_date"])) # impute missing values with oldest date
        .assign(last_review_recency = lambda df_: abs((threshs["newest_date"] - df_.last_review).dt.days)) # compute recency of last review (use absolute value to deal with dates newer than newest_date)
        .assign(last_review_recency_log_noise = lambda df_: np.log1p(df_.last_review_recency) + np.random.normal(scale = 0.1, size = len(df_))) # use log and add some noise for smoother results (helps convergence of some algos)
        # column "reviews_per_month" (impute missing vals)
        .assign(reviews_per_month = lambda df_ : df_.reviews_per_month.mask(df_.reviews_per_month.isna(), 0)) # impute missing values with 0
    )

    # drop date column (we use the new recency columns instead)
    df = df.loc[:, [col for col in df.columns if col not in ["last_review"]]]

    return df, threshs

def encode_room_type(df):
    """
    Encode room type variable (ordinal categorical variable).
    """
    room_types = ['Shared room', 'Private room', 'Entire home/apt']

    return (df
        .assign(room_type_enc = lambda df_: [room_types.index(room) for room in df_.room_type])
    )

def clean(df):
    """
    Clean dataframe after it has been merged (utility function for get_official_geo_hierarchies).
    """
    df = df.drop(["index_right"], axis = 1)
    return df

def load_official_geo_hierarchies(df):
    """
    Get official geographical hierarchies from public data of NYC state.
    ATTENTION: this function cannot be run twice! need to restart kernel!
    """
    
    # usage: .to_crs(crs = ##)
    COORD_CALC = "EPSG:3857" # metric
    COORD_PLOT = "EPSG:4326" # geodetic
    SEARCH_DIST = 1000 # (meters)
    
    # transform our dataframe to geopandas dataframe
    df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude), crs = COORD_PLOT).to_crs(crs = COORD_CALC)
    
    # drop neighbourhood & neighbourhood_group columns
    df.drop(columns = ["neighbourhood", "neighbourhood_group"], inplace = True)

    # get lvl1 (boroughs) data
    lvl1 = pd.read_csv('data/2020_NYC_Boroughs.csv')
    lvl1 = (lvl1
        .rename(columns = {"BoroCode": "GeoID", "BoroName": "name"}, inplace = False)
        .assign(geometry = lambda df_: df_.geometry.apply(wkt.loads))
    )
    lvl1 = gpd.GeoDataFrame(lvl1, geometry = "geometry", crs = COORD_PLOT).to_crs(crs = COORD_CALC)
    
    # get lvl2 (community districts) data
    lvl2 = pd.read_csv('data/2020_NYC_Community Districts.csv')
    lvl2 = (lvl2
        .rename(columns = {"CDTAName": "name"}, inplace = False)
        .assign(name = lambda df_: [n[5:].split("(")[0] if "(" in n else n[5:] for n in df_.name])
        .assign(GeoID = lambda df_: [int(str(b_code) + cd_code[2:]) for b_code, cd_code in zip(df_.BoroCode, df_.CDTA2020)])
        .assign(geometry = lambda df_: df_.geometry.apply(wkt.loads))
    )
    lvl2 = gpd.GeoDataFrame(lvl2, geometry = "geometry", crs = COORD_PLOT).to_crs(crs = COORD_CALC)
    
    # get lvl3 (neighbourhoods) data
    lvl3 = pd.read_csv('data/2020_NYC_Neighbourhoods.csv')
    lvl3 = (lvl3
        .rename(columns = {"NTA2020": "GeoID", "NTAName": "name"}, inplace = False)
        .assign(geometry = lambda df_: df_.geometry.apply(wkt.loads))
    )
    lvl3 = gpd.GeoDataFrame(lvl3, geometry = "geometry", crs = COORD_PLOT).to_crs(crs = COORD_CALC)
    
    # get lvl4 (census tracts) data
    lvl4 = pd.read_csv('data/2020_NYC_Census_Tracts.csv')
    lvl4 = (lvl4
        .rename(columns = {"GEOID": "GeoID", "CDTANAME": "name"}, inplace = False)
        .assign(name = lambda df_: [n[5:].split("(")[0] if "(" in n else n[5:] for n in df_.name])
        .assign(geometry = lambda df_: df_.geometry.apply(wkt.loads))
    )
    lvl4 = gpd.GeoDataFrame(lvl4, geometry = "geometry", crs = COORD_PLOT).to_crs(crs = COORD_CALC)
    
    # map corresponding cluster [lvl2, lvl3, lvl4] to original data
    # we use sjoin_nearest to ensure that even if borders of NYC geometry objects have some small holes, we still find
    # the "closest" true level (e.g. if airbnb listing coordinates are like 10 meters away from the border of the nearest 
    # cluster)

    for lvl, lvl_data in zip(["lvl1", "lvl2", "lvl3", "lvl4"], [lvl1, lvl2, lvl3, lvl4]):
        lvl_data_renamed = lvl_data.rename(columns = {"GeoID": lvl + "_GeoID", "name": lvl + "_name"}, inplace = False)
        df = clean(df.sjoin_nearest(lvl_data_renamed[["geometry", lvl + "_GeoID", lvl + "_name"]], how = "inner", max_distance = SEARCH_DIST))
    
    # go back to geodetic coordinate system (we had to choose metric for calculations in .sjoin_nearest)
    df   = df.to_crs(crs = COORD_PLOT)
    lvl1 = lvl1.to_crs(crs = COORD_PLOT)
    lvl2 = lvl2.to_crs(crs = COORD_PLOT)
    lvl3 = lvl3.to_crs(crs = COORD_PLOT)
    lvl4 = lvl4.to_crs(crs = COORD_PLOT)
    
    return df.sort_index(), lvl1, lvl2, lvl3, lvl4

def load_official_geo_hierarchies_data(df, lvl1, lvl2, lvl3, lvl4):
    """
    Fetch public information (e.g. population, nbr of housing units, etc.) based on assigned official hierarchies.
    """
    # load data from census 2020 (population, housing units, share of vacant housing units)
    census_data = pd.read_excel('data/2020_NYC_Censusdata_Pop_Housing.xlsx', sheet_name = 1, header = 3)
    
    # subset to the columns that are relevant for this analysis
    relevant_cols = ["GeoID", "Pop_20", "Pop_Ch", "HUnits_20", "HUnits_Ch", "VacHU_20P", "VacHU_PCh"]
    census_data = census_data[relevant_cols]
    
    # merge with airbnb dataset as well with geopandas dataset of respective level
    lvl_dfs = [lvl1, lvl2, lvl3, lvl4]
    for i, (lvl, lvl_data) in enumerate(zip(["lvl1", "lvl2", "lvl3", "lvl4"], lvl_dfs)):
        
        counts = dict(df[lvl + "_GeoID"].value_counts())
        
        df = (df
            .reset_index().merge(census_data, left_on = lvl + "_GeoID", right_on = "GeoID").set_index(df.index.names)
            .assign(listings_count = lambda df_: [counts.get(geoid, 0) for geoid in df_[lvl + "_GeoID"]])
            .assign(listings_count_norm = lambda df_: [c/pop*1000 if pop > 500 else 0 for c, pop in zip(df_.listings_count, df_.Pop_20)])
            .drop(columns = "GeoID", inplace = False)
            .rename(columns = {col: lvl + "_" + col for col in relevant_cols[1:] + ["listings_count", "listings_count_norm"]}, inplace = False)
        )
        
        if lvl_data is not None:
            lvl_dfs[i] = (lvl_data
                .merge(census_data, left_on = "GeoID", right_on = "GeoID")
                .assign(listings_count = lambda df_: [counts.get(geoid, 0) for geoid in df_.GeoID])
                .assign(listings_count_norm = lambda df_: [c/pop*1000 if pop > 500 else 0 for c, pop in zip(df_.listings_count, df_.Pop_20)])
            )
    
    # load data from American Community Survey (unemployment rate, commuting time to work,
    # nbr households, % household social security, % households retirement, household income, median earnings of workers)
    acs_data = pd.read_excel('data/20162020_NYC_Censusdata_Econ.xlsx', sheet_name = 0, header = 0)
    
    # subset to the cols that are relevant for this analysis
    relevant_cols = ["GeoID", "CvLFUEm1P", "MnTrvTmE", "HH2E", "Inc_SoSecP", "Inc_RtrmtP", "MnHHIncE", "MdEWrkE"]
    acs_data = acs_data[relevant_cols]
    
    # merge with airbnb dataset (can only do this on lvl3!) as well as with geopandas dataset of lvl3
    df = (df
        .reset_index().merge(acs_data, left_on = "lvl3_GeoID", right_on = "GeoID").set_index(df.index.names)
        .drop(columns = "GeoID", inplace = False)
        .rename(columns = {col: "lvl3_" + col for col in relevant_cols[1:]}, inplace = False)
    )
    
    lvl_dfs[2] = (lvl_dfs[2]
        .merge(acs_data, left_on = "GeoID", right_on = "GeoID")
    )
    
    lvl1, lvl2, lvl3, lvl4 = lvl_dfs[0], lvl_dfs[1], lvl_dfs[2], lvl_dfs[3]
    
    return df.sort_index(), lvl1, lvl2, lvl3, lvl4

def impute_missing_geo_vals(df, lvl1, lvl2, lvl3, lvl4, threshs = None):
    
    # useful later
    lvl_dict = {"lvl1": lvl1, "lvl2": lvl2, "lvl3": lvl3, "lvl4": lvl4}
    prev_lvl_name_dict = {"lvl2": ["lvl1"], "lvl3": ["lvl2", "lvl1"], "lvl4": ["lvl3", "lvl2", "lvl1"]}
        
    # impute missing values with value of level above
    census_cols = ["Pop_20", "Pop_Ch", "HUnits_20", "HUnits_Ch", "VacHU_20P", "VacHU_PCh"]

    # impute missing values with global average of same column
    acs_cols = ["CvLFUEm1P", "MnTrvTmE", "HH2E", "Inc_SoSecP", "Inc_RtrmtP", "MnHHIncE", "MdEWrkE"]
    
    # initialise threshs (to process test data in same way as training data)
    if threshs is None:
        threshs = dict()
        
        for col in acs_cols:
            col = "lvl3_" + col
            threshs[col] = df[col].mean()
    
    # get all cols with missing vals
    cols_missing_vals = []
    for col, missing_vals in dict(df.isna().sum()).items():
        if missing_vals > 0 and col != "name":
            cols_missing_vals.append(col)
    
    # handle missing values
    for col in cols_missing_vals:
        col_name = col[5:]
        
        if col_name in census_cols:
            prev_lvl_names = prev_lvl_name_dict[col[:4]]
            for prev_lvl_name in prev_lvl_names:
                prev_lvl = dict(lvl_dict[prev_lvl_name].set_index("GeoID")[col_name])
                df[col] = [prev_lvl[geoid] if np.isnan(val) else val for val, geoid in zip(df[col], df[prev_lvl_name + "_GeoID"])]
        
        elif col_name in acs_cols:
            df[col].fillna(threshs[col], inplace = True)
            
    return df, threshs

def encode_target(df, threshs = None):
    """
    Compute Empirical Bayes Target Encoding based on the geographical hierarchies within the dataset.

    For more details on these features, please look at the Preprocessing notebook! Everything explained there, including the math!

    If this function is applied to the validation / test data, we need to provide it all thresholds
    that we used to preprocess our training data.
    """
    fts = ["global", "lvl1_GeoID", "lvl2_GeoID", "lvl3_GeoID", "lvl4_GeoID"]

    if threshs is None:
        threshs = dict()

        for ft in fts:
            if ft == "global":
                threshs[ft] = { "count": df.price.count(),
                                "mean":  df.price.mean(),
                                "sd":    df.price.std(),
                                "var":   df.price.var()}
            else:
                threshs[ft] = { "count":    dict(df.groupby([ft])["price"].count()),
                                "mean":     dict(df.groupby([ft])["price"].mean()),
                                "sd":       dict(df.groupby([ft])["price"].std()),
                                "var":      dict(df.groupby([ft])["price"].var())
                }

    def get_stat(df_, cur_lvl, agg):
        return (df_
            .assign(temp = lambda df_: [threshs[cur_lvl][agg].get(lvl, np.nan) for lvl in df_[cur_lvl]])
            .temp
        )

    def get_lambdas(df_, cur_lvl, prev_lvl):
        return (df_
            .assign(temp = lambda df_: [(cur_count * prev_var) / (cur_var + (cur_count * prev_var)) for cur_count, cur_var, prev_var in zip(df_[cur_lvl+"_c"], df_[cur_lvl+"_v"], df_[prev_lvl+"_v"])])
            .temp
        )

    def blend_stat(df_, cur_lvl, prev_lvl, stat):
        return (df_
            .assign(lambdas = lambda df_: get_lambdas(df_, cur_lvl = cur_lvl, prev_lvl = prev_lvl))
            .assign(temp = lambda df_: [lam * cur_stat + (1 - lam) * prev_stat for lam, cur_stat, prev_stat in zip(df_.lambdas, df_[cur_lvl+"_"+stat], df_[prev_lvl+"_"+stat])])
            .temp
        )

    df = (df
        # (1) compute basic stats for each level and fall back to previous level if not available
        ## level 1
        .assign(temp1_c = lambda df_: [val if not np.isnan(val) else threshs[fts[0]]["count"] for val in get_stat(df_, cur_lvl = fts[1], agg = "count")])
        .assign(temp1_m = lambda df_: [val if not np.isnan(val) else threshs[fts[0]]["mean"]  for val in get_stat(df_, cur_lvl = fts[1], agg = "mean")])
        .assign(temp1_s = lambda df_: [val if (not np.isnan(val)) and (val > 0) else threshs[fts[0]]["sd"]  for val in get_stat(df_, cur_lvl = fts[1], agg = "sd")])
        .assign(temp1_v = lambda df_: [val if (not np.isnan(val)) and (val > 0) else threshs[fts[0]]["var"] for val in get_stat(df_, cur_lvl = fts[1], agg = "var")])
        ## level 2
        .assign(temp2_c = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[2], agg = "count"), df_.temp1_c)])
        .assign(temp2_m = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[2], agg = "mean"),  df_.temp1_m)])
        .assign(temp2_s = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[2], agg = "sd"),  df_.temp1_s)])
        .assign(temp2_v = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[2], agg = "var"), df_.temp1_v)])
        ## level 3
        .assign(temp3_c = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[3], agg = "count"), df_.temp2_c)])
        .assign(temp3_m = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[3], agg = "mean"),  df_.temp2_m)])
        .assign(temp3_s = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[3], agg = "sd"),  df_.temp2_s)])
        .assign(temp3_v = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[3], agg = "var"), df_.temp2_v)])
        ## level 4
        .assign(temp4_c = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[4], agg = "count"), df_.temp3_c)])
        .assign(temp4_m = lambda df_: [val if not np.isnan(val) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[4], agg = "mean"),  df_.temp3_m)])
        .assign(temp4_s = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[4], agg = "sd"),  df_.temp3_s)])
        .assign(temp4_v = lambda df_: [val if (not np.isnan(val)) and (val > 0) else temp for val, temp in zip(get_stat(df_, cur_lvl = fts[4], agg = "var"), df_.temp3_v)])

        # (2) compute Empirical Bayes Target Encoding
        ## blend layers 2 and 1
        .assign(temp2_m = lambda df_: blend_stat(df_, cur_lvl = "temp2", prev_lvl = "temp1", stat = "m"))
        .assign(temp2_s = lambda df_: blend_stat(df_, cur_lvl = "temp2", prev_lvl = "temp1", stat = "s"))

        ## blend layers 3 and 2
        .assign(temp3_m = lambda df_: blend_stat(df_, cur_lvl = "temp3", prev_lvl = "temp2", stat = "m"))
        .assign(temp3_s = lambda df_: blend_stat(df_, cur_lvl = "temp3", prev_lvl = "temp2", stat = "s"))

        ## blend layers 4 and 3
        .assign(temp4_m = lambda df_: blend_stat(df_, cur_lvl = "temp4", prev_lvl = "temp3", stat = "m"))
        .assign(temp4_s = lambda df_: blend_stat(df_, cur_lvl = "temp4", prev_lvl = "temp3", stat = "s"))
    )

    return (df
        # remove useless columns (cannot do this in previous step)
        .loc[:, [col for col in df.columns if col not in ["temp1_c", "temp1_m", "temp1_s", "temp1_v", "temp2_c", "temp2_v", "temp3_c", "temp3_v", "temp4_c", "temp4_v"]]]
        # update column names
        .rename(columns = {"temp2_m": "lvl2_target_m", "temp2_s": "lvl2_target_s",
                           "temp3_m": "lvl3_target_m", "temp3_s": "lvl3_target_s",
                           "temp4_m": "lvl4_target_m", "temp4_s": "lvl4_target_s"}, inplace = False)
        ), threshs

def load_data():
    """
    Load dataset including geofeatures
    """
    df = pd.read_csv('data/AB_NYC_2019.csv', index_col=0)
    df, lvl1, lvl2, lvl3, lvl4 = load_official_geo_hierarchies(df)
    df, lvl1, lvl2, lvl3, lvl4 = load_official_geo_hierarchies_data(df, lvl1, lvl2, lvl3, lvl4)
    return df, lvl1, lvl2, lvl3, lvl4

def prep_pipeline(df, lvl1, lvl2, lvl3, lvl4, drop_cols = None, impute_threshs = None, impute_geo_threshs = None, encode_threshs = None):
    """
    Preprocess the provided pandas dataframe and drop the columns provided in a Python list to drop_cat_cols.
    If this function is applied to the validation / test data, we need to provide it all thresholds
    that we used to preprocess our training data. 
    """
    df = df.copy()
    
    df, impute_threshs     = impute_missing_vals(df, threshs = impute_threshs)
    df, impute_geo_threshs = impute_missing_geo_vals(df, lvl1, lvl2, lvl3, lvl4, threshs = impute_geo_threshs)
    df                     = encode_room_type(df)
    df, encode_threshs     = encode_target(df, threshs = encode_threshs)

    if drop_cols:
        df = df.loc[:, [col for col in df.columns if col not in drop_cols]]

    return df, impute_threshs, impute_geo_threshs, encode_threshs

def split_frame(df):
    # split into X and y
    y = df.loc[:, "price"]
    X = df.copy()
    X.drop(["price"], axis = 1, inplace = True)
    return X, y