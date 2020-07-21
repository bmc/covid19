#!/usr/bin/env bash

JHU_REPO_NAME="COVID-19"
JHU_REPO="CSSEGISandData/$JHU_REPO_NAME"
#JHU_GH_REPO="git@github.com:$JHU_REPO.git"
JHU_GH_REPO="https://github.com/$JHU_REPO.git"

NYT_REPO_NAME="covid-19-data"
NYT_REPO="nytimes/$NYT_REPO_NAME"
#NYT_GH_REPO="git@github.com:$NYT_REPO.git"
NYT_GH_REPO="https://github.com/$NYT_REPO.git"

JHU="data/johns-hopkins/$JHU_REPO_NAME"
NYT="data/nytimes/$NYT_REPO_NAME"

CDC_URL="https://data.cdc.gov/api/views/r8kw-7aab/rows.csv?accessType=DOWNLOAD"
CDC_OUT="data/cdc/Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv"

update_from_git() {
  repo=$2
  dir=$1
  here=`pwd`

  echo ""
  if [ -d $dir ]
  then
      cd $dir
      echo "Updating $dir from $repo"
      git pull origin master
  else
      parent=$(dirname $dir)
      mkdir -p $parent
      cd $parent
      echo "Cloning $repo in $parent"
      git clone $repo
  fi
  status=$?
  cd $here
  return $status
}

find_command() {
  cmd=${1?'Missing command to find'}

  # Bash array syntax is fugly. See
  # https://stackoverflow.com/questions/10586153/split-string-into-an-array-in-bash

  # Split $PATH into an array.
  IFS=: read -r -a _path <<< "$PATH"

  # Loop over each element, looking for the command.
  for dir in "${_path[@]}"
  do
    if [ ! -d $dir ]
    then
      continue
    fi

    if [ -x $dir/$cmd ]
    then
      echo "$dir/$cmd"
      return 0
    fi
  done
  echo ""
  return 1
}

# Use curl or wget, whichever is available.

curl_path=$(find_command curl)
wget_path=$(find_command wget)

web_download() {
  url=${1?'Missing URL'}
  out=${2?'Missing output file'}

  echo ""
  if [ -n "$curl_path" ]
  then
    echo "Getting CDC data with curl..."
    curl -o $CDC_OUT "$CDC_URL"
    if [ $? != 0 ]
    then
      echo "Download failed!" >&2
      exit 1
    fi
  elif [ -n "$wget_path" ]
  then
    echo "Getting CDC data with wget..."
    wget -O $CDC_OUT "$CDC_URL"
    if [ $? != 0 ]
    then
      echo "Download failed!" >&2
      exit 1
    fi
  else
    echo "Can't find curl or wget." >&2
    exit 1
  fi
}
  
update_from_git data/johns-hopkins/$JHU_REPO_NAME $JHU_GH_REPO
update_from_git data/nytimes/$NYT_REPO_NAME $NYT_GH_REPO
web_download $CDC_URL "$CDC_OUT"
