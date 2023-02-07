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

def encode_geo_fts(df, threshs = None):
    """
    Compute Euclidian distance to neighbourhood_group and neighbourhood center and 
    compute Empirical Bayes Target Encoding based on the provided dataframe.
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

    df = (df
        # (1) distances to centers
        # compute distance (euclidian) to center of neighbourhood_group
        .assign(distance_l1 = lambda df_: [((lat - threshs[fts[1]]["lat_mean"][lvl])**2 + (lon - threshs[fts[1]]["lon_mean"][lvl])**2) ** 0.5 for lat, lon, lvl in zip(df_.latitude, df_.longitude, df_.neighbourhood_group)])
        # compute distance (euclidian) to center of neighbourhood
        .assign(distance_l2 = lambda df_: [((lat - threshs[fts[2]]["lat_mean"][lvl])**2 + (lon - threshs[fts[2]]["lon_mean"][lvl])**2) ** 0.5 for lat, lon, lvl in zip(df_.latitude, df_.longitude, df_.neighbourhood)])
        # if unknown region (after train-val-test split some level 2 entries might not be included in training data), impute with previous level encoding
        .assign(distance_l2 = lambda df_: df_.distance_l2.mask(df_.distance_l2.isna(), df_.distance_l1))
        
        # (2) target encoding
        # compute blending factor between level 0 (global) and 1 (neighbourhood_group)
        .assign(numerator = lambda df_: [(threshs[fts[1]]["count"][lvl1] * threshs[fts[0]]["var"]) for lvl1 in df_.neighbourhood_group])
        .assign(lambda_l1 = lambda df_: [num / (threshs[fts[1]]["var"][lvl1] + num) for num, lvl1 in zip(df_.numerator, df_.neighbourhood_group)])
        # compute blending factor between level 1 (neighbourhood_group) and 2 (neighbourhood)
        .assign(numerator = lambda df_: [(threshs[fts[2]]["count"][lvl2] * threshs[fts[1]]["var"][lvl1]) for lvl1, lvl2 in zip(df_.neighbourhood_group, df_.neighbourhood)])
        .assign(lambda_l2 = lambda df_: [num / (threshs[fts[2]]["var"][lvl2] + num) for num, lvl2 in zip(df_.numerator, df_.neighbourhood)])
        # compute target encoding of mean and sd between level 0 (global) and 1 (neighbourhood_group)
        .assign(l1_mean = lambda df_: [lam * threshs[fts[1]]["mean"][lvl1] + (1 - lam) * threshs[fts[0]]["mean"] for lam, lvl1 in zip(df_.lambda_l1, df_.neighbourhood_group)])
        .assign(l1_sd   = lambda df_: [lam * threshs[fts[1]]["sd"][lvl1]   + (1 - lam) * threshs[fts[0]]["sd"]   for lam, lvl1 in zip(df_.lambda_l1, df_.neighbourhood_group)])
        # compute target encoding of mean and sd between level 1 (neighbourhood_group) and 2 (neighbourhood) & add a bit of noise for better convergence
        .assign(l2_mean = lambda df_: [lam * threshs[fts[2]]["mean"][lvl2] + (1 - lam) * l1_m + np.random.normal()  for lam, lvl2, l1_m  in zip(df_.lambda_l2, df_.neighbourhood, df_.l1_mean)])
        .assign(l2_sd   = lambda df_: [lam * threshs[fts[2]]["sd"][lvl2]   + (1 - lam) * l1_sd + np.random.normal(scale = 0.5) for lam, lvl2, l1_sd in zip(df_.lambda_l2, df_.neighbourhood, df_.l1_sd)])
        # if unknown region (after train-val-test split some level 2 entries might not be included in training data), impute with previous level encoding
        .assign(l2_mean = lambda df_: df_.l2_mean.mask(df_.l2_mean.isna(), df_.l1_mean))
        .assign(l2_sd   = lambda df_: df_.l2_sd.mask(df_.l2_sd.isna(), df_.l1_sd))
    )

    return (df
        # remove useless columns ["numerator", "lambda_l1", "lambda_l2", "l1_mean", "l1_sd"] (cannot do this in previous step)
        .loc[:, [col for col in df.columns if col not in ["numerator", "lambda_l1", "lambda_l2", "l1_mean", "l1_sd"]]]
        ), threshs

def prep_pipeline(df, impute_threshs = None, encode_threshs = None):
    """
    Preprocess the provided pandas dataframe.
    If this function is applied to the validation / test data, we need to provide it all thresholds
    that we used to preprocess our training data. 
    """

    df, impute_threshs = impute_missing_vals(df, threshs = impute_threshs)
    df, encode_threshs = encode_geo_fts(df, threshs = encode_threshs)

    return df, impute_threshs, encode_threshs