#%%
import pandas as pd
import seaborn
from utils import *

rvs = [
    work_contract_type,
    profile_years_experience,
    salary_in_usd,
    salary_monthly_NETO,
    tools_programming_language
]

assert all(rv in DB.columns.values.tolist() for rv in rvs)

df = DB[
    (DB.work_contract_type == "Full-Time") &
    (DB.salary_monthly_NETO > 1000) &
    (DB.profile_years_experience <= 5)
][rvs]

cured_programming_languages = "cured_programming_languages"
programming_language = "programming_language"

df = add_cured_col(
    df=df,
    cured_col=cured_programming_languages,
    uncured_col=tools_programming_language,
    cure_func=split_languages
)

df = stack_col(
    df=df,
    stacked_col=programming_language,
    unstacked_col=cured_programming_languages
)
#%%

# Distribución de personas por Lenguaje
print(df[programming_language].value_counts())


#%%
# Calculo de media del salario por lenguaje
print(df)
df.groupby(programming_language).mean().sort_values(
    by=salary_monthly_NETO,
    ascending=False
)


# %%
