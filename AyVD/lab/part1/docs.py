# %% [markdown]
# # Diplomatura en Ciencas de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matias Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo

# %% [markdown]
# Inicilamente definiremos algunas funciones, constantes y nombres de variables
# que utilizaremos durante nuestro análisis.

# %%
import pandas as pd
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

URL = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/sysarmy_survey_2020_processed.csv"
DB = pd.read_csv(URL)

MINWAGE_IN_ARG = 18600

work_contract_type = "work_contract_type"
profile_years_experience = "profile_years_experience"
salary_in_usd = "salary_in_usd"
salary_monthly_NETO = "salary_monthly_NETO"
tools_programming_language = "tools_programming_languages"
salary_monthly_BRUTO = "salary_monthly_BRUTO"
work_province = "work_province"
profile_age = "profile_age"
profile_gender = "profile_gender"

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


def clean_outliers_bygroup(df, by, column_name):
    pass

def clean_outliers(dataset, column_name):
    """Returns dataset removing the outlier rows from column @column_name."""
    interesting_col = dataset[column_name]
    # Here we can remove the outliers from both ends, or even add more restrictions.
    mask_outlier = (
        np.abs(interesting_col - interesting_col.mean()) <= (2.5 * interesting_col.std()))
    return dataset[mask_outlier]


def to_categorical(column, bin_size=10, min_cut=0, max_cut=50):
    if min_cut is None:
        min_cut = int(round(column.min())) - 1
    value_max = int(np.ceil(column.max()))
    max_cut = min(max_cut, value_max)
    intervals = [(x, x + bin_size) for x in range(min_cut, max_cut, bin_size)]
    if max_cut != value_max:
        intervals.append((max_cut, value_max))
    print(intervals)
    return pd.cut(column, pd.IntervalIndex.from_tuples(intervals))


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
# Random variables defined in utils.py
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
df_langs = df[df[programming_language].isin(best_langs.index.to_list())].reset_index(drop=True)
df_langs[[programming_language, salary_monthly_NETO]] \
    .groupby(programming_language) \
    .describe()
# %% [markdown]
# Antes de visualizar como distribuyen los salarios nos interesará eliminar los
# outliers para cada lenguaje. Esto se realiza por medio de `clean_outliers`
# removiendo para cada lenguaje, los salarios que estén a una distancia 2.5
# veces su desvio estandar.
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
# Notar que el 25% de los empleados que utilizan **Go** cobran a lo sumo
# \$71000 de salario neto y el 75% \$106000, posicionandose como el lenguaje
# mejor pago, llegando hasta un máximo de \$150000 mensuales! También puede
# verse que en su mayoría el minimo salario está cerca del vital y móvil
# decretado por el país (\$18000). También los lenguajes **Javascript**,
# **HTML**, **.NET**, y **CSS** otorgan salarios similares, ¿Distribuiran de
# manera similar? Los
# siguientes boxenplots muestran dicha similitud.
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
# ## Asociación
# Para ver si existe una correlación entre el salario bruto y el neto analizamos
# el coeficiente de correlación de Pearson $\rho$ entre estas variables aleatorias.

# %%
df = DB.copy()

salary_cols = [salary_monthly_NETO, salary_monthly_BRUTO]
df[salary_cols].corr()

# %% [markdown]
# Notar que el valor de $\rho$ entre los salarios nos dá un valor positivo
# cercano a 1. Esto nos indica que existe una correlación entre las variables
# que se comporta aproximadamente lineal pero que aún así podría aún haber una
# fuerte relación no lineal entre ellas. También podemos visualizar la
# distribución conjunta de estas variables por medio un `scatterplot`.

# %%
seaborn.scatterplot(
    data=df[salary_cols],
    x=salary_monthly_BRUTO, y=salary_monthly_NETO
)
plt.axvline(df[salary_monthly_BRUTO].mean(), color="black", linestyle="--", label="mean")
plt.axhline(df[salary_monthly_NETO].mean(), color="black", linestyle="--")
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.legend()

# %% [markdown]
# Claramente concuerda con el valor del $\rho$, sin embargo también deja en
# evidencia que un valor cerca de 1 no necesariamente implica que al incrementar
# el valor de una variable causa que la otra incremente. Observamos que una gran
# cantidad de sueldos en bruto están asociados con otra numerosa cantidad de
# sueldos en neto, debido a ello, podríamos prescindir del salario bruto.
# %% [markdown]
# ## Densidad Conjunta
# El análisis de la Densidad Conjunta, en un primer momento nos permite detectar
# si existe algún patrón o compartamiento determinado entre dos variables, es
# decir si una variable se ve afectada ante un cambio en la otra. Si vemos que
# existe una cierta dependencia entre ellas, podemos incluso modelar una función
# de densidad o probalidad conjunta.
#
# Para variables numericas es util calcular medidas como la Covarianza o el
# Coeficiente de Correlación Lineal de Pearson, y tambien usar gráficos como por
# ejemplo de dispersión o de lineas. En cambio para variables categóricas es más
# frecuente usar como visualizaciones las tablas de frecuencias relativas o
# gráficos de barras. Cuando analizamos una variable categorica y una numerica
# son comunes los gráficos de barra o gráficos de caja.

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
# fuerte como se observa en el siguiente gráfico de dispersión.
# %%
seaborn.pairplot(
    data=df,
    y_vars=[profile_years_experience],
    x_vars=[salary_monthly_NETO],
    height=4,aspect=2
)
plt.ticklabel_format(style='plain', axis='x')
# %% [markdown]
# Observamos que la variable años de experiencia es discreta, por lo cual
# optamos por generar rangos para poder interpretar mejor la relación de estas
# variables.
# %%
profile_years_segment = 'profile_years_segment'

df[profile_years_segment] = to_categorical(
    df[profile_years_experience],
    min_cut=0,
    max_cut=30,
    bin_size=10
)
# %% [markdown]
# Con el siguiente gráfico podemos visualizar mejor la relación entre años de
# experiencia y salario mensual neto. Podemos observar que los salarios netos en
# promedio rondan en los \$115000 para el rango de 10 a 30 años de experiencia.
# Sin embargo, para el rango comprendido entre 0 y 10 años de experiencia y entre
# 30 y 44 años de experiencia el salario promedio es menor.
# 
# Notar que para los primeros 10 años de experiencia el intervalo de confianza
# es mucho más chico debido a que su estimación está dada por una muestra más
# representativa que para los intervalos (10, 20], (20, 30] y (30, 44]. Siendo
# este último el menos confiable. Ahora bien, es claro ver que el salario neto
# medio aumenta a partir de los primeros 10 años de experiencia. Esto lo
# confirma el siguiente cuadro de medidas descriptivas
# %%

fig = plt.figure(figsize=(8,6))
seaborn.barplot(
    data=df,
    estimator=np.mean,
    x=profile_years_segment,
    y=salary_monthly_NETO
)
plt.xticks(rotation=45)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Años de experiencia")
plt.ticklabel_format(style='plain', axis='y')
# %%
df[[profile_years_segment, salary_monthly_NETO]] \
    .groupby(profile_years_segment) \
    .describe() \
    .sort_values(by=profile_years_segment, ascending=True)

# %% [markdown]
# ## Análisis Provincias de Argentina y Salario Neto
# Para una mejor visualización de este análisis decidimos agrupar las provincias
# por regiones y de esta forma obtener grupos más representativos.

# %%
# TODO: Is there better way?
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

region = "region"

df[region] = df[work_province].replace(new_groups)
fig = plt.figure(figsize=(8,6))
seaborn.barplot(
    y=df[salary_monthly_NETO],
    x=df[region],
    estimator=np.mean, 
    order=order
)
plt.xticks(rotation=90)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Zonas de Argentina")
plt.ticklabel_format(style='plain', axis='y')
# %%
df[[region, salary_monthly_NETO]].groupby(region).describe()
# %% [markdown]
# Conluimos que las regiones de Buenos Aires y Centro, son las mejores pagas,
# siendo Noreste y Nordeste las que tienen el menor salario neto promedio.

# %% [markdown]
# ## Años de Edad - Tipos de Contrato

# %%

profile_age_segment = "profile_age_segment"

df[profile_age_segment] = to_categorical(
    df[profile_age],
    bin_size=5,
    min_cut=15,
    max_cut=50
)

fig = plt.figure(figsize=(10,10))
df_ages_fulltime = df[df[work_contract_type] == "Full-Time"] \
    .groupby(profile_age_segment).size() \
    .to_frame().rename(columns={0: "count"}) \
    .reset_index()
df_ages_nofulltime = df[df[work_contract_type] != "Full-Time"] \
    .groupby(profile_age_segment).size() \
    .to_frame().rename(columns={0: "count"}) \
    .reset_index()

seaborn.pointplot(
    data=df_ages_fulltime,
    x=profile_age_segment, y="count",
    color='b',
    legend='fulltime'
)

seaborn.pointplot(
    data=df_ages_nofulltime,
    x=profile_age_segment, y="count",
    color='r',
    legend='nofulltime'
)
plt.legend()
plt.show()

# %% [markdown]
# Concluimos que la edad sí tiene una influencia en la elección del tipo de
# contrato. Notar
#

# %% [markdown]
# ## Tipos de Contraros Y Salarios Netos Medios

# %%
new_groups = {
    'Part-Time':'Otros contratos',
    'Tercerizado (trabajo a través de consultora o agencia)':'Otros contratos',
    'Remoto (empresa de otro país)':'Otros contratos',
    'Freelance':'Otros contratos'}
grouped_contract = df[work_contract_type].replace(new_groups)
seaborn.barplot(y=df[salary_monthly_NETO], x=grouped_contract, estimator=np.mean)
plt.xticks(rotation=55)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Tipos de contracto")
plt.ticklabel_format(style='plain', axis='y')

# %% [markdown]
# Observamos que la media de los otros contratos es considerablemente superior
# que los Full - Time. Veamos especificamente en que tipo de contrato se
# encuentran estos mayores sueldos.

# %%
seaborn.barplot(
    y=df[salary_monthly_NETO],
    x=df[work_contract_type],
    estimator=np.mean
)
plt.xticks(rotation=90)
plt.ylabel("Media de salario mensual NETO")
plt.xlabel("Tipos de contracto")
plt.ticklabel_format(style='plain', axis='y')

# %% [markdown]
# Los mayores sueldos se encuentran en las personas que trabajan de forma Remota
# para otro país, es decir los sueldos dolarizados.

# %%
df[[work_contract_type, salary_monthly_NETO]] \
    .groupby(work_contract_type) \
    .describe()

# %% [markdown]
# ## Densidad condicional
# Estudie la distribución del salario según el nivel de estudio.
# Separe la población según el nivel de estudio (elija dos subpoblaciones
# numerosas) y grafique de manera comparativa ambos histogramas de la variable
# 'salary_monthly_NETO' ¿Considera que ambas variables son independientes? ¿Qué
# analizaría al respecto?
# Calcule medidas de centralización y dispersión para cada subpoblación
# %% [markdown]
# ## Distribución del salario neto y nivel de estudio

# %%
df = DB[(DB[salary_monthly_NETO] > MINWAGE_IN_ARG)]
df = clean_outliers(df, salary_monthly_NETO)

seaborn.catplot(data=df, y='salary_monthly_NETO',
                x='profile_studies_level', height=4, aspect=2)

Study_count = df.profile_studies_level.value_counts()\
    .reset_index()\
    .rename(columns={'index': 'Study Level', 'profile_studies_level':'Frecuency'})
Study_count[:10]
# %%
salary_col= 'salary_monthly_NETO'
df_U = df[df['profile_studies_level'] == 'Universitario']
df_T =df[df['profile_studies_level'] == 'Terciario']

plt.hist(df_U[salary_col], color='red', bins=50)
plt.hist(df_T[salary_col], color='blue', bins=50)
plt.show()
# %% [markdown]
# La probabilidad de estar por arriba del promedio sin importar el grado de
# estudio 33,13%, mientras que la probabilidad de estar por arriba del promedio
# teniendo un nivel de estudio terciario es de el 23,88 %
# %%
avg_salary = df[salary_monthly_NETO].mean()

p_above_avg = len(df[df[salary_col] >= avg_salary]) / len(df)

is_above_avg = len(df[df[salary_col] > avg_salary])#Cantidad A
Terciario= len(df[df["profile_studies_level"] == "Terciario"]) #Cantidad B
Total=len(df) #Cantidad Total
condicion= (df[salary_col] > avg_salary) & (df["profile_studies_level"] == "Terciario")
Prob_AB=len(df[condicion])/Terciario

is_above_avg2 = len(df[df[salary_col] > avg_salary])#Cantidad A
Universitario = len(df[df["profile_studies_level"] == "Universitario"]) #Cantidad B
Total2=len(df) #Cantidad Total
condicion2= (df[salary_col] > avg_salary) & (df["profile_studies_level"] == "Universitario")
Prob_AB2=len(df[condicion2])/Universitario
Prob_AB2

print(
    f"La probabilidad de estar por arriba del promedio sin importar el grado de estudio es {p_above_avg * 100:.2f}%"
)
print(
    f"mientras que la probabilidad de estar por arriba del promedio \
    teniendo un nivel de estudio terciario es de el {Prob_AB * 100:.2f}%"
)
print(
    f"Teniendo estudios universitarios es {Prob_AB2 * 100:.2f}%"
)

# %%

Prob_A= is_above_avg/Total
Prob_B= Terciario/Total

(round(Prob_A * Prob_B, 4), round(Prob_AB, 4))
# %%
(round(Prob_AB, 4), round(Prob_A, 4))

# %% [markdown]
# Densidad Conjunto Condicional

rvs = [
    work_province,
    work_contract_type,
    salary_monthly_NETO,
    profile_years_experience,
    profile_gender
]

df = DB[
    (DB[profile_years_experience] < 50) &
    (DB[profile_age] < 100) &
    (DB[salary_monthly_NETO] > MINWAGE_IN_ARG) &
    (DB.salary_in_usd != "Mi sueldo está dolarizado")
][rvs]

df = clean_outliers(df, salary_monthly_NETO)
df.describe().round(2)

# %%

plt.figure(figsize=(10,6))
seaborn.scatterplot(data=df.sample(1000), 
    x=profile_years_experience, y=salary_monthly_NETO,
    marker='.',
    hue=profile_gender
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# %%
