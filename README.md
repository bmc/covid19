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

- [CDC Provisional Death Counts for Coronavirus Disease](https://www.cdc.gov/nchs/nvss/vsrr/covid19/index.htm).
  Before running the CDC notebook, the data file (see tree, below) must be downloaded from the CDC link 
  and copied into the appropriate data location.
- [Johns Hopkins University's Center for Systems Science and Engineering (CSSE) COVID-19 data repository](https://github.com/CSSEGISandData/COVID-19).
  Before running the Johns Hopkins notebook, the linked GitHub repository must be cloned under `data/johns-hopkins`.
- [Data from The New York Times, based on reports from state and local health agencies](https://github.com/nytimes/covid-19-data)
  Before running the New York Times notebook, the linked GitHub repository must be cloned under `data/nytimes`.
- [State FIPS codes](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/?cid=nrcs143_013696) from the USDA. Stored in `data/states-fips.csv` and
  checked into this repository.
- [County FIPS codes](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697) from the USDA. Stored in
  `data/state-county-fips.csv` and checked into this repository.
- [World Population Review](https://worldpopulationreview.com/states/) (for state population data). Stored in `state-populations.csv` and
  checked into this repository.

Run `./update.sh` to clone and update the Johns Hopkins and New York Times
data sources. It will create them if they don't exist and update them if they
do.

## Data Layout

They assume the existence of a `data` subdirectory. Items marked with an
asterisk (`*`) are checked into this repo. Others must be created, either
via downloading the required files or cloning a GitHub repository.

```
data/
  |-- states-fips.csv (*)
  |-- state-populations.csv (*)
  |-- state-county-fips.csv (*)
  |
  +-- cdc/
  |    |
  |    +--- Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv
  |
  +-- johns-hopkins/
  |    |
  |    +--- COVID-19/
  |
  +-- nytimes/
       |
       +--- covid-19-data
```

- The `data/johns-hopkins/COVID-19` folder is assumed to be a cloned copy
  of <https://github.com/CSSEGISandData/COVID-19>.
- The CSV file in the `cdc` folder is assumed to be a download from the
  CDC page referenced above.
