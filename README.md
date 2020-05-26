This repo contains Jupyter notebooks that I use to play with COVID-19
data.

## Jupyter setup

I've been running these notebooks using the `jupyter/scipy-notebook`
Docker image.

I run the image as follows:

```shell
$ docker run -v $HOME:/home/jovyan/$USER -p 127.0.0.1:9999:8888 jupyter/scipy-notebook jupyter notebook --NotebookApp.token= --NotebookApp.password=
```

You can add `-d` if you want to run it as a daemon, but I just run it under
`screen` or `tmux`.

That command line:

- Allows me to connect my browser to http://localhost:9999/lab to get to
  Jupyter.
- Ensures that Docker _only_ binds to localhost, not to any of the external
  interfaces.
- Disables the password and token, so I don't have to type a password or
  token when I connect my browser to Jupyter.
- Mounts my entire home directory into the Jupyter image, under
  `/home/jovyan/bmc` (in my case). Thus, I have full access to my
  home directory from the Jupyter browser interface.

## Data Sources

The notebooks currently use data from these data sources:

Data sources:

- [CDC Provisional Death Counts for Coronavirus Disease](https://www.cdc.gov/nchs/nvss/vsrr/covid19/index.htm)
- [Johns Hopkins University's Center for Systems Science and Engineering (CSSE) COVID-19 data repository](https://github.com/CSSEGISandData/COVID-19)
- [World Population Review](https://worldpopulationreview.com/states/) (for state population data).

## Data Layout

They assume the existence of a (not-checked-in) `data` subdirectory.
Here's the required layout:

```
data/
  |
  +-- cdc/
  |    |
  |    +--- Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv
  |
  +-- johns-hopkins/
       |
       +--- COVID-19/
```

- The `data/johns-hopkins/COVID-19` folder is assumed to be a cloned copy
  of <https://github.com/CSSEGISandData/COVID-19>.
- The CSV file in the `cdc` folder is assumed to be a download from the
  CDC page referenced above.
