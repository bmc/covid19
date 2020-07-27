# COVID-19 notebooks

This repo contains [Jupyter](https://jupyter.org/) notebooks that I use to
analyze some COVID-19 data. You're welcome to play with these notebooks, adapt 
them, copy them, etc. 

This repo is main just a playground allowing me to experiment with the data,
as well as  bone up on Jupyter, [`matplotlib`](https://matplotlib.org/), and
[Pandas](https://pandas.pydata.org/). 

I will not claim to be a Pandas or `matplotlib` expert. What you see is what
you get. (I also welcome suggestions for improvement.)

I generally check the notebooks into Git _with_ the results intact.
That means you can browse to a notebook in GitHub, and it should render
the notebook and its results. (GitHub can render Jupyter notebooks for
display only.)

## The code

### Notebooks

- `cdc.ipynb`: Notebook that analyzes data from the Centers for Disease
  Control and Prevention
- `johns-hopkins.ipynb`: Notebook that analyzes data from the Johns Hopkins
  University's Center for Systems Science and Engineering (CSSE)
- `nytimes.ipynb`: Notebook that analyzes data from the New York Times
- `update-data.ipynb`: Notebook that can be used to download updated data.
  See below for details.

### Other code

- `lib/common.py`: Some common Python code shared between the notebooks and
  imported into each.

## License


The notebooks and code in this repository are released under a
[Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA) license.

This license does _not_ apply to any of the data, including the data that
is cached in this repository.

## A note about the generated graphs

The various notebooks use Pandas and `matplotlib` to graph COVID-19 data. 
In most cases, the code also saves each graph in a PNG file in an `images` 
subdirectory. `images` is not checked in; the notebooks create the directory
if it does not exist.

Saving the graphs as images makes them easy to share.


## Jupyter setup

I've been running these notebooks using my own `bclapper/jupyter-scipy-plus`
Docker image. This image is a child of the stock `jupyter/scipy-notebook`
Docker image, but it prebuilds the following Jupyter extensions (so I don't
have to build and install them every time I fire up a new Docker image):

- [jupyter-lab-go-to-definition](https://github.com/krassowski/jupyterlab-go-to-definition)
- [jupyterlab-python-file](https://github.com/jtpio/jupyterlab-python-file)

`bclapper/jupyter-scipy-plus` is pushed to DockerHub, so you can just fire it
up (or pull it down, whatever). If you want to see how it's been built, go
[here](https://github.com/bmc/docker/tree/master/jupyter-scipy-plus) and
take a look at `Dockerfile`.

If you don't need these extra extensions, feel free to run the base 
`jupyter/scipy-notebook` Docker image. The notebooks will work fine with
either image.

I run the image as follows:

```shell
$ docker run -v $HOME:/home/jovyan/$USER -p 127.0.0.1:9999:8888 bclapper/jupyter-scipy-plus jupyter notebook --NotebookApp.token= --NotebookApp.password=
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

## Data

### Data Sources

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

### Getting the Data

#### Johns Hopkins, New York Times, and CDC data

To get the Johns Hopkins, New York Times, and CDC data, you can either run
`./update-data.sh` from a shell, or you can run the `update-data.ipynb`
notebook within Jupyter (which just runs `./update-data.sh`).

Either way, the update process:

- clones the Johns Hopkins and New York Times GitHub repos, if they don't
  exist already; or updates them, if they do
- downloads the latest CDC data (as CSV) from the CDC web site.


#### CDC data

If you're going to run the `cdc.ipynb` notebook, you have to download the CDC
data manually. Go to the CDC page listed above, download the data as a CSV
file, and copy that file to
`data/cdc/Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv`.

### Data Layout

The notebooks assume the existence of a `data` subdirectory. Items marked
with an asterisk (`*`) are checked into this repo. Others must be created,
either via downloading the required files or cloning some GitHub repositories.

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
  of Johns Hopkins repo listed above.
- The `data/nytimes/covid-19-data` folder is assumed to be a cloned copy
  of New York Times repo listed above.
- The CSV file in the `cdc` folder is assumed to be a download from the
  CDC page referenced above.

See the previous section for instructions on obtaining the Johns Hopkins,
New York Times, and CDC data.

