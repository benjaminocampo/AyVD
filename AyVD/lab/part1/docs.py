"""
Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones

Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
"""

"""
# Analisis descriptivo

Para la respuesta a la pregunta **¿Cuáles son los lenguajes de programación
asociados a los mejores salarios?** hay varios factores a tener en cuenta además
del salario neto de los empleados, tales como, el tipo de contrato (Full Time o
Part Time), si su sueldo está dolarizado o no, los años de experiencia, su rol
de trabajo, entre otras que pueden dificultar el analisis de la pregunta si no
nos enfocamos en una subpoblación del total de trabajadores. Por lo tanto, para
hacer una comparación más justa optamos por considerar aquellos empleados que
cumplan lo siguiente:

- Contrato Full Time: Dependiendo de la cantidad de horas que se empeñe en el
  rol el salario podría verse influenciado. Por ende consideramos solamente el
  de mayor carga horaria.
- A lo sumo 5 años de experiencia: Nos interesará saber que lenguajes otorgan
  los mejores salarios en los primeros años de trabajo.
- Un salario neto mayor al minimo vital y móvil en la Argentina: Filtraremos
  además aquellos empleados que no superen el salario mínimo para empleados
  mensualizados.

Esto llevó a la selección de las siguientes variables aleatorias relevantes
`rvs`.
"""

#%%
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
    (DB.profile_years_experience <= 5)
][rvs]


"""
Agregaremos una columna adicional `programming_language` que apile sobre cada
empleado sus lenguajes de programación utilizados dado por
`tools_programming_languages`
"""
#%%

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

"""
## Lenguajes de Programación más Populares

Para cada lenguaje de programación obtenemos el conteo de su frecuencia junto a
su salario neto promedio.
"""
#%%
# TODO: Can we avoid the join clause using aggregations?
gb = df.groupby(programming_language)
count_bylangs = gb.agg({salary_monthly_NETO: "mean"}) \
    .join(gb.size().to_frame().rename(columns={0: "count"})) \

count_bylangs.sort_values(by=salary_monthly_NETO, ascending=False).head(30)

"""
Notar que si bien hay algunos lenguajes que tienen sueldos muy altos, su
frecuencia es muy baja, siendo muy poco representativo. Por el contrario si
ordenamos por frecuencia obtenemos:
"""
# %%
count_bylangs.sort_values(by="count", ascending=False).head(30)


"""
Si analizamos la tendencia central de la frecuencia podremos determinar que
cantidad es representantiva para determinar si un lenguaje es popular o no. Para
ello, tomamos umbrales $\in{1,..100}$ calculando la media y mediana para los
lenguajes de programación que tengan una frecuencia mayor a cada uno de esos
umbrales.
"""
#%%

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

"""
Puede verse a través de la gráfica para los distintos umbrales cuales son los
valores de la media y mediana después del filtro, dando como menor distancia
para el umbral marcado en lineas punteadas. Por ende tomamos el frame resultante
de utilizar este umbral.
"""
# %%
best_langs = count_bylangs[count_bylangs["count"] >= best_threshold].sort_values(by="count", ascending=False)
best_langs

"""


"""


"""
## Selección de le
"""
# %%
