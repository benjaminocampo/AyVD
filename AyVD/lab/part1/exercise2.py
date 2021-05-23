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
salary_monthly_BRUTO = "salary_monthly_BRUTO"
work_contract_type = "work_contract_type"
salary_in_usd = "salary_in_usd"
work_province = "work_province"
profile_age = "profile_age"
profile_gender = "profile_gender"

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