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
from seaborn.miscplot import palplot
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
df_langs = clean_outliers_bygroup(
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
# mejor pago, llegando hasta un máximo de \$150000 mensuales! También puede
# verse que en su mayoría el minimo salario está cerca del vital y móvil
# decretado por el país. También los lenguajes **Javascript**, **HTML**, y
# **CSS** otorgan salarios similares, ¿Distribuiran de manera similar? Los
# siguientes boxenplots mustran dicha similitud.
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
# ligeramente lo supera, sin embargo tenemos una asimetría o sesgo hacia los
# valores más chicos de la distribución. Lo cual nos diría que, si bien es el
# lenguaje mayor pago, un poco más del 50% tienen un salario menor que los
# \$100000. A pesar de este sesgo es el único de los 12 que no presenta grandes
# asimetrias en sus colas y cuyos valores extremos no son tan altos. Notar como
# Javascript, HTML, .NET y CSS distribuyen de manera similar. Esto nos daría
# como trabajo a futuro revisar si esto se mantiene para distintos roles de los
# empleados dentro de la compañia. Por último los boxplots dejan en evidencia su
# desventaja al no tener información sobre el primer y el cuarto cuantiles.

# %% [markdown]
# ## Asociación
# Para ver si existe una correlación entre el salario bruto y el neto analizamos
# el coeficiente de correlación de Pearson $\rho$ entre estas variables aleatorias.

# %%
salary_cols = [salary_monthly_NETO, salary_monthly_BRUTO]
DB[salary_cols].corr()

# %% [markdown]
# Notar que el valor de $\rho$ entre los salarios nos dá un valor positivo
# cercano a 1. Esto nos indica que existe una correlación entre las variables
# que se comporta aproximadamente lineal pero que aún así podría aún haber una
# fuerte relación no lineal entre ellas. También podemos visualizar la
# distribución conjunta de estas variables por medio un `scatterplot`.

# %%
seaborn.scatterplot(data=DB[salary_cols], x=salary_monthly_BRUTO, y=salary_monthly_NETO)
plt.axvline(DB[salary_monthly_BRUTO].mean(), color="black", linestyle="--", label="mean")
plt.axhline(DB[salary_monthly_NETO].mean(), color="black", linestyle="--")
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.legend()

# %% [markdown]
# Claramente concuerda con el valor del $\rho$, sin embargo también deja en
# evidencia que un valor cerca de 1 no necesariamente implica que al incrementar
# el valor de una variable causa que la otra incremente. Lo único que nos dice
# es que una grán cantidad de sueldos en bruto están asociados con otra numerosa
# cantidad de sueldos en neto. Pero esto no ocurre para valores que están
# realmente alejados de la media de ambas variables. Si bien podrían ser
# considerados outliers, otros más cercanos siguen este comportamiento que
# dejarían en cuestión si realmente puede considerarse eliminar la columna de
# salario bruto.

# %% [markdown]
# TODO: Cambiar la conclusión para poner que se puede sacar la columna de bruto

# %% [markdown]
# ## Densidad Conjunta
# 1) El análisis de la Densidad Conjunta, en un primer momento nos permite
#    detectar si existe algún patrón o compartamiento determinado entre dos
#    variables, es decir si una variable se ve afectada ante un cambio en la
#    otra. Si vemos que existe una cierta dependencia entre ellas, podemos
#    incluso modelar una función de densidad o probalidad conjunta.
# 
#    Para variables numericas es util calcular medidas como la Covarianza o el
#    Coeficiente de Correlación Lineal de Pearson, y tambien usar gráficos como
#    por ejemplo de dispersión o de lineas. En cambio para variables categóricas
#    es más frecuente usar como visualizaciones las tablas de frecuencias
#    relativas o gráficos de barras. Cuando analizamos una variable categorica y
#    una numerica son comunes los gráficos de barra o gráficos de caja.

# %% [markdown]
# ## Elección de las Variables
# Categóricas: `work_province`, `work_contract_type`
#
# Numéricas: `salary_monthly_NETO`, `profile_years_experience`, `profile_age`
#

# %%
rvs = [
    work_province,
    work_contract_type,
    salary_monthly_NETO,
    profile_years_experience,
    profile_age
]

df = DB[
    (DB[profile_years_experience] < 50) &
    (DB[profile_age] < 100) &
    (DB[salary_monthly_NETO] > MINWAGE_IN_ARG)
][rvs]

df = clean_outliers(df, salary_monthly_NETO)
df.describe().round(2)
# %% [markdown]
# ## Análisis de Años de Experiencia y Salario Neto
# %%
df[[salary_monthly_NETO, profile_years_experience]].corr()
# %% [markdown]
# Se observa que $\rho$ entre los años de experiencia y el salario neto es
# positivo pero próximo a 0. Lo cual significa que no tienen una relación lineal
# fuerte como se observa en el siguioente gráfico de dispersión.
# %%
seaborn.pairplot(
    data=df,
    y_vars=[profile_years_experience],
    x_vars=[salary_monthly_NETO],
    height=4,aspect=2
)
plt.ticklabel_format(style='plain', axis='x')

# %%
plt.figure(figsize=(10,6))
seaborn.lineplot(
    data=df,
    x=profile_years_experience, y=salary_monthly_NETO,
    estimator= 'mean' , ci=None
)
# %% [markdown]
# Observamos que la variable años de experiencia, al ser discreta, optamos por
# generar rangos para poder interpretar mejor la relación de estas variables. A
# pesar de los picos, notar que existe una tendencia creciente para los primeros
# 15 años de experiencia, pero luego dejan de observarse un patrón determinado.

df['profile_years_segment'] = to_categorical(df[profile_years_experience], max_cut=30)
# %% [markdown]
# Con el siguiente gráfico podemos visualizar mejor la relación entre años de
# experiencia y salario mensual neto. Podemos observar que los salarios netos en
# promedio rondan en los 125000 para el rango de 10 a 40 años de experiencia,
# sin embargo para el rango comprendido entre 0 y 10 años de experiencia y entre
# 40 y 50 años de experiencia el salario promedio baja.

# Notar que para los primeros 10 años de experiencia el intervalo de confianza
# es mucho más chico debido a que su estimación está dada por una muestra más
# representativa que para los intervalos (10, 20], (20, 30] y (30, 44]. Siendo
# este último el menos confiable. Ahora bien, es claro ver que el salario neto
# medio aumenta a partir de los primeros 10 años de experiencia. Esto lo
# confirma el siguiente cuadro de medidas descriptivas
# %%

fig = plt.figure(figsize=(8,6))
seaborn.barplot(data=df, x='profile_years_segment', y='salary_monthly_NETO', estimator=np.mean)
plt.xticks(rotation=45)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Años de experiencia")
plt.ticklabel_format(style='plain', axis='y')
# %%
group_col = 'profile_years_segment'
Tabla_1= df[[group_col, salary_monthly_NETO]].groupby(group_col).describe().sort_values(by="profile_years_segment",ascending=True)
Tabla_1
# %% [markdown]
# ## ANÁLISIS PROVINCIAS DE ARGENTINA Y SALARIO NETO

# %%
fig = plt.figure(figsize=(20,6))
seaborn.barplot(y=df[salary_monthly_NETO], x=df[work_province], estimator=np.mean)
plt.xticks(rotation=90)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Zonas de Argentina")
plt.ticklabel_format(style='plain', axis='y')
# %% [markdown]
# No es claro visualizar este gráfico con tantas categorías y además vemos que
# existen ciertas provincias con muy pocos valores, procedemos a agrupar por
# zonas.

# %%

new_groups = {'Jujuy':'Nordeste y Noreste',
'Salta':'Nordeste y Noreste',
'Tucumán':'Nordeste y Noreste',
'Catamarca':'Nordeste y Noreste',
'Santiago del Estero':'Nordeste y Noreste',
'La Rioja':'Nordeste y Noreste',
'Corrientes':'Nordeste y Noreste',
'Entre Ríos':'Nordeste y Noreste',
'Chaco':'Nordeste y Noreste',
'Misiones':'Nordeste y Noreste',
'Formosa':'Nordeste y Noreste',
'GBA':'Buenos Aires',
'Provincia de Buenos Aires':'Buenos Aires',
'Córdoba':'Centro',
'Santa Fe':'Centro',
'La Pampa':'Centro',
'Santiago del Estero':'Centro',
'San Luis':'Cuyo y Patagonia',
'Mendoza':'Cuyo y Patagonia',
'San Juan':'Cuyo y Patagonia',
'Tierra del Fuego':'Cuyo y Patagonia',
'Santa Cruz':'Cuyo y Patagonia',
'Río Negro':'Cuyo y Patagonia',
'Chubut':'Cuyo y Patagonia',
'Neuquén':'Cuyo y Patagonia'}
order = ['Nordeste y Noreste', 'Centro', 'Buenos Aires','Ciudad Autónoma de Buenos Aires', 'Cuyo y Patagonia']
df["grouped_province"] = df[work_province].replace(new_groups)
fig = plt.figure(figsize=(8,6))
seaborn.barplot(y=df[salary_monthly_NETO], x=df["grouped_province"], estimator=np.mean, 
                                order=order
                )
plt.xticks(rotation=90)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Zonas de Argentina")
plt.ticklabel_format(style='plain', axis='y')
# %%
df[["grouped_province", salary_monthly_NETO]].groupby("grouped_province").describe()
# %% [markdown]
# Podemos observar que para las regiones del Centro, Buenos Aires y Cuyo y
# Patagonia el salario medio es similar y cercano al salario medio del total de
# la muestra. Sin embargo, para la región del Nordeste y Noreste es ligeramente
# más baja.

# %% [markdown]
# ## 3. Años de Edad - Tipos de Contrato

# %%

df["profile_age_segment"] = to_categorical(df[profile_age], bin_size=5, min_cut=15, max_cut=50)

fig = plt.figure(figsize=(10,10))
df_ages_fulltime = df[df[work_contract_type] == "Full-Time"] \
    .groupby("profile_age_segment").size() \
    .to_frame().rename(columns={0: "count"}) \
    .reset_index()
df_ages_nofulltime = df[df[work_contract_type] != "Full-Time"] \
    .groupby("profile_age_segment").size() \
    .to_frame().rename(columns={0: "count"}) \
    .reset_index()

seaborn.pointplot(
    data=df_ages_fulltime,
    x="profile_age_segment", y="count",
    color='b',
    legend='fulltime'
)

seaborn.pointplot(
    data=df_ages_nofulltime,
    x="profile_age_segment", y="count",
    color='r',
    legend='nofulltime'
)
plt.legend()
plt.show()
# %% [markdown]
# ## 4. Tipos de Contraros Y Salarios Netos Medios

# %%
new_groups = {'Part-Time':'Otros contratos',
'Tercerizado (trabajo a través de consultora o agencia)':'Otros contratos',
'Remoto (empresa de otro país)':'Otros contratos',
'Freelance':'Otros contratos'}
grouped_contract = df.work_contract_type.replace(new_groups)
seaborn.barplot(y=df['salary_monthly_NETO'], x=grouped_contract, estimator=np.mean)
plt.xticks(rotation=55)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Tipos de contracto")
plt.ticklabel_format(style='plain', axis='y')

# %% [markdown]
# Observamos que la media de los otros contratos en considerablemente superior
# que los Full - Time. Veamos especificamente en que tipo de contrato se
# encuentran esos mayores sueldo

# %%
seaborn.barplot(y=df['salary_monthly_NETO'], x=df['work_contract_type'], estimator=np.mean)
plt.xticks(rotation=90)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Tipos de contracto")
plt.ticklabel_format(style='plain', axis='y')

# %% [markdown]
# Los mayores sueldos se encuentran en las personas que trabajan de forma Remota
# para otro país, es decir los sueldos dolarizados.

# %%
group_col = 'work_contract_type'
Tabla_2= df[[group_col,"salary_monthly_NETO"]].groupby(group_col).describe()
Tabla_2
# %%
