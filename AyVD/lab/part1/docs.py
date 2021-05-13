# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo


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
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from utils import *

# Random variables defined in utils.py
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
    (DB.salary_monthly_NETO > MINWAGE_IN_ARG) &
    (DB.profile_years_experience <= 5) &
    (DB.salary_in_usd != "Mi sueldo está dolarizado")   # TODO: this column should be cleaned before
][rvs]

# %% [markdown]
# También agregaremos una columna adicional `cured_programming_language`
# utilizando la función `add_cured_col` donde a cada uno de los lenguajes de
# programación dados por la columna `tools_programming_language`, se los separa
# en listas de python. Luego apilamos sobre cada empleado sus lenguajes de
# programación utilizados dados por la columna curada utilizando la función
# `stack_col` para obtener el siguiente dataframe.
# %%

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

df = df.reset_index(drop=True)

df

# %% [markdown]
# ## Lenguajes de Programación más Populares
# Para cada lenguaje de programación obtenemos el conteo de su frecuencia junto
# a su salario neto promedio.
#%%
# TODO: Can we avoid the join clause using aggregations?
gb = df.groupby(programming_language)
count_bylangs = gb.agg(salary_monthly_NETO_mean=(salary_monthly_NETO, "mean")) \
    .join(gb.size().to_frame().rename(columns={0: "count"})) \

count_bylangs.sort_values(by="salary_monthly_NETO_mean", ascending=False).head(20)

# %% [markdown]
# Notar que si bien hay algunos lenguajes que tienen sueldos muy altos, su
# frecuencia es muy baja, siendo muy poco representativo. Por el contrario si
# ordenamos por frecuencia obtenemos:
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
tendency_df, best_threshold = min_central_tendency(count_bylangs, "count", 100)

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
# para el umbral marcado en lineas punteadas. También deja en evidencia que
# aquellos sin remover los lenguajes pocos usados la media y la mediana de los
# conteos resultan ser muy distantes.
# %%
best_langs = count_bylangs[count_bylangs["count"] >= best_threshold]
total_count = best_langs["count"].sum()
best_langs = add_cured_col(
    best_langs,
    cured_col="percentage",
    uncured_col="count",
    cure_func=lambda count: count / total_count * 100
)
best_langs.sort_values(by="salary_monthly_NETO_mean", ascending=False)

# %% [markdown]
# Notar que del total de empleados que utilizan estos 12 lenguajes, el 1.9%
# tienen el mejor salario neto promedio y utilizan **Go** para programar. Luego
# les siguen **Python** con el 8.6% y **Java** con el 9.9%. Notar que en el caso
# de **Javascript** es el más popular con un 18% y un salario no muy alejado de
# los primeros puestos.

# %% [markdown]
# ## Distribución de salario por lenguaje
# Utilizaremos la lista de lenguajes anterior para obtener la distribución de
# salarios por cada uno de ellos para el dataset filtrado por las condiciones
# iniciales.
# %%
df_langs = df[df[programming_language].isin(best_langs.index.to_list())]
df_langs

# %% [markdown]
# Antes de visualizar como distribuyen los salarios nos interesará eliminar los
# outliers para cada lenguaje. Esto se realiza por medio de `clean_outliers`
# removiendo para cada lenguaje, los salarios que estén a una distancia 2.5
# veces su desvio estandar.
# %%
df_langs = clean_outliers(
    df_langs,
    by=programming_language,
    column_name=salary_monthly_NETO
)
df_langs[[programming_language, salary_monthly_NETO]] \
    .groupby(programming_language) \
    .describe()

# %% [markdown]
# Notar que el 25% de los empleados que utilizan **Go** cobran a lo sumo
# \$73120.50 de salario neto y el 75% \$106000, posicionandose como el lenguaje
# mejor pago, llegando hasta un máximo de \$150000 mensuales! También puede verse
# que en su mayoría el minimo salario está cerca del vital y móvil decretado por
# el país. También los lenguajes **Javascript**, **HTML**, y **CSS** otorgan
# salarios similares, los cuales podría deverse a que es más utilizado como
# "stack", posiblemente para un desarrollador de web frontend.
# %%
plt.figure(figsize=(12, 6))
seaborn.boxplot(data=df_langs, x=salary_monthly_NETO, y=programming_language,
                color='orangered')
plt.ticklabel_format(style='plain', axis='x')
# %%

