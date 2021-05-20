# %%
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

URL = "https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/sysarmy_survey_2020_processed.csv"
DB = pd.read_csv(URL)

# random variables
salary_monthly_NETO = "salary_monthly_NETO"
profile_gender = "profile_gender"

df = DB[[salary_monthly_NETO, profile_gender]]

df.groupby(profile_gender).describe()

alpha = 0.05

is_man = df.profile_gender == 'Hombre'

groupA = df[(df[salary_monthly_NETO] > 1000) & is_man][salary_monthly_NETO]
groupB = df[(df[salary_monthly_NETO] > 1000) & ~is_man][salary_monthly_NETO]


# %% [markdown]
# ## Estimación Puntual
# Consideramos las variables aleatorias $X_A$ y $X_B$, salario neto de los
# hombres (**groupA**), y el salario neto de las mujeres y otros (**groupB**)
# respectivamente. El estimador $\hat{\theta}$ que vamos a utilizar es $\overline{X_A} -
# \overline{X_B}$. Por lo tanto la estimación puntual obtenida es:
# %%
groupA.mean() - groupB.mean()

# %% [markdown]
# Notar que la diferencia de las medias del salario de ambos de grupos es de
# \$23262 o equivalentemente:

# %% 
(groupA.mean() - groupB.mean()) / groupA.mean() * 100
# %% [markdown]
# El grupo de hombres cobran casi un 23% más del salario neto que el conformado por las mujeres y otros.

# %% [markdown]
# Aparte de reportar la estimación puntual se calculo el error estandar como una
# medida de precisión del estimador, es decir,
# 
# $\sqrt{Var(\hat{\theta})} =
# \sqrt{Var(\overline{X_A} - \overline{X_B})} = \sqrt{Var(\overline{X_A}) +
# Var(\overline{X_B}))} =  \sqrt{\frac{\sigma_A^{2}}{n_A} +
# \frac{\sigma_B^2}{n_B}}$
#
# %%
np.sqrt(groupA.std()**2 / groupA.size + groupB.std()**2 / groupB.size)
# %% [markdown]
# Notar que la diferencia de las medias del salario de ambos de grupos es de
# \$23262 con un error de estimación de alrededor de los \$2400, es decir, que
# la estimación realizada tiene un bajo desvío, lo cual nos indicaría que tiene
# buena precisión lo cual podría deberse al gran tamaño de las muestras.

# %% [markdown]
# ## Intervalo de Confianza para $\mu_A - \mu_B$
# Ahora compararemos la estimación puntual obtenida con la de un intervalo de
# confianza cuyo nivel de significancia es de $1 - \alpha = 0.95$. Notar que el
# tamaño de la muestra para el grupo A es de 4815 y el del grupo B de 891, por
# lo tanto resultaría conveniente calcular el intervalo de confianza utilizando
# un estadistico Z.
# %%
cm = sms.CompareMeans(sms.DescrStatsW(groupA), sms.DescrStatsW(groupB))
cm.zconfint_diff(alpha=alpha, usevar='unequal')
# %% [markdown]
# Con un 95% de confianza se espera que la diferencia de salarios medios entre
# el grupo A y B este comprendido entre \$18561 y \$27964. Es decir, si
# generamos sucesivos intervalos por medio de este estadistico, el 95% de ellos
# van a contener el parámetro. Entonces hay una chance del 95% de que el
# obtenido sea uno de esos intervalos.
# 

# %%
cm.tconfint_diff(alpha=alpha, usevar='unequal')


# %%
cm.ztest_ind(alternative="larger", usevar="unequal")

# %%
