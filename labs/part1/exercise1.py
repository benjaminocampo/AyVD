# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo

# %% [markdown]
# Inicilamente definiremos algunas funciones, constantes y nombres de variables
# que utilizaremos durante nuestro análisis.

# %%
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

URL = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/sysarmy_survey_2020_processed.csv"
DB = pd.read_csv(URL)

MINWAGE_IN_ARG = 18600

profile_years_experience = "profile_years_experience"
salary_in_usd = "salary_in_usd"
salary_monthly_NETO = "salary_monthly_NETO"
tools_programming_language = "tools_programming_languages"
work_contract_type = "work_contract_type"


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

# %% [markdown]
# ## Agrupamiento por lenguages de programación
# Trabajaremos con el dataset dado por la Encuesta Sysarmy del año 2020. Ahora
# bien, antes de abordar el problema, agregaremos la columna adicional
# `cured_programming_language` a nuestro dataset utilizando la función
# `add_cured_col` donde a cada uno de los lenguajes de programación dados por la
# columna `tools_programming_language`, se los separa en listas de python. Luego
# apilamos sobre cada empleado, sus lenguajes de programación utilizados dados
# por la columna curada utilizando la función `stack_col` para obtener el
# siguiente dataframe.

# %%
cured_programming_languages = "cured_programming_languages"
programming_language = "programming_language"

df = DB.copy() \
    .pipe(
        add_cured_col,
        cured_col=cured_programming_languages,
        uncured_col=tools_programming_language,
        cure_func=split_languages
    ).pipe(
        stack_col,
        stacked_col=programming_language,
        unstacked_col=cured_programming_languages
    ).reset_index(drop=True)

df
# %% [markdown]
# ## Analisis descriptivo

# Para la respuesta a la pregunta **¿Cuáles son los lenguajes de programación
# asociados a los mejores salarios?** hay varios factores a tener en cuenta
# además del salario neto de los empleados, tales como, el tipo de contrato
# (Full Time o Part Time), si su sueldo está dolarizado o no, los años de
# experiencia, su rol de trabajo, entre otras que pueden dificultar el analisis
# de la pregunta si no nos enfocamos en una subpoblación del total de
# trabajadores. Por lo tanto, para hacer una comparación más justa optamos por
# considerar aquellos empleados que cumplan lo siguiente:
#
# - Contrato Full Time: Dependiendo de la cantidad de horas que se empeñe en el
#   rol el salario podría verse influenciado. Por ende consideramos solamente el
#   de mayor carga horaria.
# - A lo sumo 5 años de experiencia: Nos interesará saber que lenguajes otorgan
#   los mejores salarios en los primeros años de trabajo.
# - Un salario neto mayor al minimo vital y móvil en la Argentina: Filtraremos
#   además aquellos empleados que no superen el salario mínimo para empleados
#   mensualizados.
#
# Esto llevó a la selección de las siguientes variables aleatorias relevantes
# `rvs`.

# %%
rvs = [
    programming_language,
    work_contract_type,
    profile_years_experience,
    salary_in_usd,
    salary_monthly_NETO,
]

df = df[
    (df[work_contract_type] == "Full-Time") &
    (df[salary_monthly_NETO] > MINWAGE_IN_ARG) &
    (df[profile_years_experience] <= 5) &
    (df[salary_in_usd] != "Mi sueldo está dolarizado")
][rvs]

df
# %% [markdown]
# ## Lenguajes de Programación más Populares
# Para cada lenguaje de programación obtenemos el conteo de su frecuencia junto
# a su salario neto promedio.
#%%
count_bylangs = df.groupby(programming_language).agg(
    salary_monthly_NETO_mean=(salary_monthly_NETO, "mean"),
    count=(programming_language, "count")
)

count_bylangs.sort_values(by="salary_monthly_NETO_mean", ascending=False).head(20)

# %% [markdown]
# Notar que si bien hay algunos lenguajes que tienen sueldos muy altos, su
# frecuencia es muy baja, siendo muy poco representativo. Por lo tanto
# ordenaremos por frecuencia.
# %%
count_bylangs.sort_values(by="count", ascending=False).head(20)

# %% [markdown]
# Ahora bien, ¿Cuantos empleados deben utilizar el lenguaje para ser considerado
# popular?. Si analizamos la tendencia central de la frecuencia podremos
# determinar que cantidad es representantiva. Para ello, tomamos umbrales entre
# $\{1, ... , 100\}$ calculando la media y mediana para los lenguajes de
# programación que tengan una frecuencia mayor a cada uno de esos umbrales y
# seleccionando el umbral que minimice la distancia entre estas medidas. Esto
# lo hacemos a través de la función `min_central_tendency` siendo 100 el umbral
# máximo a considerar bajo el dataframe obtenido en la celda anterior.
# %%
max_threshold = 100

tendency_df, best_threshold = min_central_tendency(
    count_bylangs,
    "count",
    max_threshold
)

fig = plt.figure(figsize=(15, 5))
seaborn.lineplot(
    data=tendency_df,
    x="threshold", y="value", hue="metric"
)
plt.axvline(best_threshold, color="r", linestyle="--", label="best threshold")
plt.legend()
plt.ticklabel_format(style="plain", axis="x")
seaborn.despine()

# %% [markdown]
# Puede verse a través de la gráfica para los distintos umbrales cuales son los
# valores de la media y mediana después del filtro, dando como menor distancia
# para el umbral marcado a partir de las lineas punteadas.
# %%
best_langs = count_bylangs[count_bylangs["count"] >= best_threshold]
best_langs
# %%

best_langs.join(
    best_langs["count"] \
        .apply(lambda count: count / best_langs["count"].sum() * 100) \
        .round(2) \
        .to_frame() \
        .rename(columns={"count": "percentage"})
        
    ).sort_values(by="salary_monthly_NETO_mean", ascending=False)

# %% [markdown]
# Notar que del total de empleados que utilizan estos 12 lenguajes, el 1.89%
# tienen el mejor salario neto promedio y utilizan **Go** para programar. Luego
# les siguen **Python** con el 8.64% y **Bash/Shell y Java** con 4.80% y 9.95%
# respectivamente. Notar que en el caso de **Javascript** es el más popular con
# un 18% y un salario no muy alejado de
# los primeros puestos. 


# %% [markdown]
# ## Distribución de salario por lenguaje
# Utilizaremos la lista de lenguajes anterior, para obtener la distribución de
# salarios por cada uno de ellos para el dataset filtrado por las condiciones
# iniciales.
# %%
df_langs = df[
    df[programming_language].isin(best_langs.index.to_list())
].reset_index(drop=True)

df_langs[[programming_language, salary_monthly_NETO]] \
    .groupby(programming_language) \
    .describe()
# %% [markdown]
# Antes de visualizar como distribuyen los salarios, nos interesará eliminar los
# outliers que estén a una distancia 2.5 veces su desvio estandar por cada
# lenguage.
# %%

dff = df_langs[[programming_language, salary_monthly_NETO]] \
    .groupby(programming_language) \
    .agg(
        mean=(salary_monthly_NETO, "mean"),
        std=(salary_monthly_NETO, "std")
    )
dff["limit"] = dff["mean"] + 2.5*dff["std"]

# %%
df_langs = df_langs.merge(dff, on=programming_language)
df_langs = df_langs[df_langs[salary_monthly_NETO] <= df_langs["limit"]]
# %%

df_langs[[programming_language, salary_monthly_NETO]] \
    .groupby(programming_language) \
    .describe()

# %% [markdown]
# Notar que el 25% de los empleados que utilizan **Go** cobran a lo sumo 71000
# de salario neto y el 75% 106000, posicionandose como el lenguaje mejor pago,
# llegando hasta un máximo de 150000 mensuales! También puede verse que en su
# mayoría el minimo salario está cerca del vital y móvil decretado por el país
# al momento de la encuesta (18600). También los lenguajes **Javascript**,
# **HTML**, **.NET**, y **CSS** otorgan salarios similares, ¿Distribuiran de
# manera similar? Los siguientes boxenplots muestran dicha similitud.
# %%
similar_langs = ["html", "javascript", ".net", "css"]
plt.figure(figsize=(12, 6))
seaborn.boxenplot(
    data=df_langs[df_langs[programming_language].isin(similar_langs)],
    x=salary_monthly_NETO, y=programming_language,
    color='orangered'
)
plt.ticklabel_format(style='plain', axis='x')


plt.figure(figsize=(12, 6))
seaborn.boxenplot(
    data=df_langs,
    x=salary_monthly_NETO, y=programming_language,
    color='orangered'
)

# %% [markdown]
# Notar como el rango intercuartil para cada una de las distribuciones de
# salarios se encuentra por debajo de los \$100000 salvo para **Go** que
# ligeramente lo supera. Sin embargo, tenemos una asimetría o sesgo hacia los
# valores más chicos de la distribución. Lo cual nos diría que, si bien es el
# lenguaje mayor pago, un poco más del 50% tienen un salario menor que los
# \$100000. Notar como Javascript, HTML, .NET y CSS distribuyen de manera
# similar. Esto nos daría como trabajo a futuro revisar si esto se mantiene para
# distintos roles de los empleados dentro de la compañia. Por último los
# boxplots dejan en evidencia su desventaja al no tener información sobre el
# primer y el cuarto cuantiles.

# %% [markdown]
# ## Empleos del ¿Futuro? 🐱‍🏍
#
# Son cada vez mas las personas que sienten que sus trabajos están quedando
# estancados, obsoletos, sin proyección y con sueldos cada vez mas bajos.
#
# Data Sciense, BI, IT, Programación, Big Data, Machine Learning, son algunos de
# los términos utilizados al hablar del nuevo mercado laboral.
#
# Un requisito común que observamos en las búsquedas laborales es el manejo de
# lenguajes de programación. Hoy en día existen múltiples lenguajes para
# aprender y muchos tienen fines similares.
#
# Entonces... ¿Con que lenguaje empezar?
#
# Según la encuesta de programadores de Sysarmy, el top 10 de lenguajes mas
# populares para comenzar este desafío son los siguientes:

# %%
gb = df_langs.groupby(programming_language)
count_bylangs = gb \
    .agg(salary_monthly_NETO_mean=(salary_monthly_NETO, "mean")) \
    .join(gb.size().to_frame().rename(columns={0: "count"}))

count_bylangs.sort_values(
    by="salary_monthly_NETO_mean",
    ascending=False
).head(10)
# %%
fig = plt.figure(figsize=(8, 6))
top10 = [
    "go",
    "python",
    "bash/shell",
    "java",
    "typescript",
    "javascript",
    "sql",
    "css",
    "html",
    ".net"
]
seaborn.barplot(
    data=df_langs[df_langs[programming_language].isin(top10)],
    x='programming_language',
    y='salary_monthly_NETO',
    estimator=np.mean,
    ci=None,
    order=top10
)
plt.xticks(rotation=90)
plt.ylabel("Salario Medio")
plt.xlabel("Lenguajes")
plt.ticklabel_format(style='plain', axis='y')
