import pandas as pd
import numpy as np

URL = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/sysarmy_survey_2020_processed.csv"
DB = pd.read_csv(URL)

MINWAGE_IN_ARG = 23544


# random variables
work_contract_type = "work_contract_type"
profile_years_experience = "profile_years_experience"
salary_in_usd = "salary_in_usd"
salary_monthly_NETO = "salary_monthly_NETO"
tools_programming_language = "tools_programming_languages"


def split_languages(languages_str):
    if not isinstance(languages_str, str):
        return []

    for label in ['ninguno de los anteriores', 'ninguno']:
        languages_str = languages_str.lower().replace(label, '')

    return [lang.strip().replace(',', '') for lang in languages_str.split()]


def stack_col(df, stacked_col, unstacked_col):
    return df[unstacked_col] \
        .apply(pd.Series).stack()\
        .reset_index(level=-1, drop=True).to_frame()\
        .join(df)\
        .rename(columns={0: stacked_col})


def add_cured_col(df, uncured_col, cured_col, cure_func):
    df.loc[:, cured_col] = df[uncured_col] \
        .apply(cure_func)
    return df


def min_central_tendency(df, col, max_threshold):
    tendency = [
        (
            threshold,
            df[df[col] > threshold][col].mean(),
            df[df[col] > threshold][col].median()
        )
        for threshold in range(df[col].min(), max_threshold)
    ]

    tendency_df = pd.DataFrame(tendency, columns=['threshold', 'mean', 'median'])
    tendency_df["distance"] = abs(tendency_df["mean"] - tendency_df["median"])
    best_threshold = tendency_df.idxmin()["distance"]

    return (
        tendency_df.melt(id_vars='threshold', var_name='metric'),
        best_threshold
    )


def clean_outliers(df, by, column_name):
    subgroups = np.unique(df[by].values)
    subcol = lambda sg: df[df[by] == sg][column_name]

    masks = [
        subcol(sg) - subcol(sg).mean() <= (2.5*subcol(sg).std())
        for sg in subgroups
    ]
    mask_outliers = pd.concat(masks, ignore_index=True)

    # TODO: Use DataFrame.drop instead
    return df[df.index.isin(mask_outliers.index[mask_outliers])]
