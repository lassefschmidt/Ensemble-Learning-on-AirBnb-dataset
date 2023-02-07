import pandas as pd
import numpy as np

def impute_missing_vals(df, threshs = None):
    """
    Impute missing values in the provided pandas dataframe. If this function is applied to the validation / test data,
    we need to provide it all thresholds that we used to preprocess our training data. 
    """
    if threshs is None:
        threshs = {"oldest_date": min(pd.to_datetime(df.last_review)),
                    "newest_date": max(pd.to_datetime(df.last_review))}

    return (df
        # column "host_name" (drop)
        .loc[:, df.columns != "host_name"]
        # column "last_review" (reformat & impute missing vals)
        .assign(last_review = lambda df_: pd.to_datetime(df_.last_review)) # convert "last_review" to datetime object format
        .assign(last_review = lambda df_: df_.last_review.mask(df_.last_review.isna(), threshs["oldest_date"])) # impute missing values with oldest date
        .assign(last_review_recency = lambda df_: abs((threshs["newest_date"] - df_.last_review).dt.days)) # compute recency of last review (use absolute value to deal with dates newer than newest_date)
        .assign(last_review_recency_log_noise = lambda df_: np.log1p(df_.last_review_recency) + np.random.normal(scale = 0.1, size = len(df_))) # use log and add some noise for smoother results (helps convergence of some algos)
        # column "reviews_per_month" (impute missing vals)
        .assign(reviews_per_month = lambda df_ : df_.reviews_per_month.mask(df_.reviews_per_month.isna(), 0)) # impute missing values with 0
        ), threshs

def encode_room_type(df):
    """
    Encode room type variable (ordinal categorical variable).
    """
    room_types = ['Shared room', 'Private room', 'Entire home/apt']

    return (df
        .assign(room_type_enc = lambda df_: [room_types.index(room) for room in df_.room_type])
    )

def encode_geo_fts(df, threshs = None):
    """
    Compute Euclidian distance to neighbourhood_group and neighbourhood center and 
    compute Empirical Bayes Target Encoding based on the provided dataframe.

    For more details on these features, please look at the Preprocessing notebook! Everything explained there, including the formulas!


    If this function is applied to the validation / test data, we need to provide it all thresholds
    that we used to preprocess our training data.
    """
    fts = ["global", "neighbourhood_group", "neighbourhood"]

    if threshs is None:
        threshs = dict()

        for ft in fts:
            if ft == "global":
                threshs[ft] = { "mean": df.price.mean(),
                                "sd":   df.price.std(),
                                "var":  df.price.var()}
            else:
                threshs[ft] = { "count":    dict(df.groupby([ft])["price"].count()),
                                "mean":     dict(df.groupby([ft])["price"].mean()),
                                "sd":       dict(df.groupby([ft])["price"].std()),
                                "var":      dict(df.groupby([ft])["price"].var()),
                                "lat_mean": dict(df.groupby([ft])["latitude"].mean()),
                                "lon_mean": dict(df.groupby([ft])["longitude"].mean()),
                }

    # replace nan values in lvl2 dict with corresponding values from lvl1 (in case region is unknown, might happen in test data)
    region_mapping = dict()
    for lvl1, lvl2 in df.groupby(fts[1])[fts[2]].value_counts().index:
        region_mapping[lvl2] = lvl1
    for name, subdict in threshs[fts[2]].items():
        for lvl2, value in subdict.items():
            if np.isnan(value):
                subdict[lvl2] = threshs[fts[1]][name][region_mapping[lvl2]] 

    df = (df
        # (1) distances to centers (for lvl2: if region unknown [might happen in test data], impute with lvl1 encoding)
        # compute distance (euclidian) to center of neighbourhood_group
        .assign(distance_l1 = lambda df_: [((lat - threshs[fts[1]]["lat_mean"].get(lvl, None))**2 + (lon - threshs[fts[1]]["lon_mean"].get(lvl, None))**2) ** 0.5 for lat, lon, lvl in zip(df_.latitude, df_.longitude, df_.neighbourhood_group)])
        # compute distance (euclidian) to center of neighbourhood
        .assign(distance_l2 = lambda df_: [((lat - threshs[fts[2]]["lat_mean"].get(lvl2, threshs[fts[1]]["lat_mean"][lvl1]))**2 + (lon - threshs[fts[2]]["lon_mean"].get(lvl2, threshs[fts[1]]["lon_mean"][lvl1]))**2) ** 0.5 for lat, lon, lvl1, lvl2 in zip(df_.latitude, df_.longitude, df_.neighbourhood_group, df_.neighbourhood)])
        
        # (2) target encoding (for lvl2: if region unknown [might happen in test data], impute with lvl1 encoding)
        # compute blending factor between level 0 (global) and 1 (neighbourhood_group)
        .assign(numerator = lambda df_: [(threshs[fts[1]]["count"].get(lvl1, None) * threshs[fts[0]]["var"]) for lvl1 in df_.neighbourhood_group])
        .assign(lambda_l1 = lambda df_: [num / (threshs[fts[1]]["var"].get(lvl1, None) + num) for num, lvl1 in zip(df_.numerator, df_.neighbourhood_group)])
        # compute blending factor between level 1 (neighbourhood_group) and 2 (neighbourhood)
        .assign(numerator = lambda df_: [(threshs[fts[2]]["count"].get(lvl2, threshs[fts[1]]["count"][lvl1]) * threshs[fts[1]]["var"].get(lvl1, None)) for lvl1, lvl2 in zip(df_.neighbourhood_group, df_.neighbourhood)])
        .assign(lambda_l2 = lambda df_: [num / (threshs[fts[2]]["var"].get(lvl2, threshs[fts[1]]["var"][lvl1]) + num) for num, lvl1, lvl2 in zip(df_.numerator, df_.neighbourhood_group, df_.neighbourhood)])
        # compute target encoding of mean and sd between level 0 (global) and 1 (neighbourhood_group)
        .assign(l1_mean = lambda df_: [lam * threshs[fts[1]]["mean"].get(lvl1, None) + (1 - lam) * threshs[fts[0]]["mean"] for lam, lvl1 in zip(df_.lambda_l1, df_.neighbourhood_group)])
        .assign(l1_sd   = lambda df_: [lam * threshs[fts[1]]["sd"].get(lvl1, None)   + (1 - lam) * threshs[fts[0]]["sd"]   for lam, lvl1 in zip(df_.lambda_l1, df_.neighbourhood_group)])
        # compute target encoding of mean and sd between level 1 (neighbourhood_group) and 2 (neighbourhood) & add a bit of noise for better convergence
        .assign(l2_mean = lambda df_: [lam * threshs[fts[2]]["mean"].get(lvl2, threshs[fts[1]]["mean"][lvl1]) + (1 - lam) * l1_m + np.random.normal()  for lam, lvl1, lvl2, l1_m  in zip(df_.lambda_l2, df_.neighbourhood_group, df_.neighbourhood, df_.l1_mean)])
        .assign(l2_sd   = lambda df_: [lam * threshs[fts[2]]["sd"].get(  lvl2, threshs[fts[1]]["sd"][lvl1])   + (1 - lam) * l1_sd + np.random.normal(scale = 0.5) for lam, lvl1, lvl2, l1_sd in zip(df_.lambda_l2, df_.neighbourhood_group, df_.neighbourhood, df_.l1_sd)])
    )

    return (df
        # remove useless columns ["numerator", "lambda_l1", "lambda_l2", "l1_mean", "l1_sd"] (cannot do this in previous step)
        .loc[:, [col for col in df.columns if col not in ["numerator", "lambda_l1", "lambda_l2", "l1_mean", "l1_sd"]]]
        ), threshs

def prep_pipeline(df, drop_cols = None, impute_threshs = None, encode_threshs = None):
    """
    Preprocess the provided pandas dataframe and drop the columns provided in a Python list to drop_cat_cols.
    If this function is applied to the validation / test data, we need to provide it all thresholds
    that we used to preprocess our training data. 
    """
    df = df.copy()

    df, impute_threshs = impute_missing_vals(df, threshs = impute_threshs)
    df                 = encode_room_type(df)
    df, encode_threshs = encode_geo_fts(df, threshs = encode_threshs)

    if drop_cols:
        df = df.loc[:, [col for col in df.columns if col not in drop_cols]]

    return df, impute_threshs, encode_threshs

def split_frame(df):
    # split into X and y
    y = df.loc[:, "price"]
    X = df.copy()
    X.drop(["price"], axis = 1, inplace = True)
    return X, y