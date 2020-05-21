This repo contains Jupyter notebooks that I use to play with COVID-19
data.

## Jupyter setup

I've been running these notebooks using the `jupyter/scipy-notebook`
Docker image.

## Data Sources

The notebooks currently use data from these data sources:

Data sources:

- [CDC Provisional Death Counts for Coronavirus Disease](https://www.cdc.gov/nchs/nvss/vsrr/covid19/index.htm)
- [Johns Hopkins University's Center for Systems Science and Engineering (CSSE) COVID-19 data repository](https://github.com/CSSEGISandData/COVID-19)

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
