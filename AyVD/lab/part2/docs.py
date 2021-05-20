# %%
import numpy as np
import pandas as pd

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
sigma = np.sqrt(groupA.std()**2 / groupA.size + groupB.std()**2 / groupB.size)
# %% [markdown]
# Notar que la diferencia de las medias del salario de ambos de grupos es de
# \$23262 con un error de estimación de alrededor de los \$2400, es decir, que
# la estimación realizada tiene un bajo desvío, lo cual nos indicaría que tiene
# buena precisión lo cual podría deberse al gran tamaño de las muestras
# %%

left = groupA.mean() - groupB.mean() - 1.96 * sigma
right = groupA.mean() - groupB.mean() + 1.96 * sigma
(left, right)

# %%
