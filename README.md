# How to run the directories for each subject

First you need to get installed `conda` which is a python package manager, and `jupyter lab`. For an step
by step guide see the inscructions in the conda user guide.

After having installed `conda` create a virtual environment for the subject
you are working on. For example, if you want to run the notebooks on the directory AyVD:

1. Create the environment from the environment.yml file:

```
$ conda diploDatos-ayvd create -f environment.yml
```

2. Activate the environment for having available the packages included in `environment.yml`

```
$ conda activate diplodatos-ayvd
```

3. Run jupyter lab in the main directory

```
$ jupyter lab
```

4. Open the jupyter notebook you're interesed in analyzing inside the subject directory.
