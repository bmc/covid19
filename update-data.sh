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

update() {
  repo=$2
  dir=$1
  here=`pwd`

  if [ -d $dir ]
  then
      cd $dir
      echo "Updating $dir"
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

update data/johns-hopkins/$JHU_REPO_NAME $JHU_GH_REPO
update data/nytimes/$NYT_REPO_NAME $NYT_GH_REPO

#here=`pwd`
#for dir in data/johns-hopkins/COVID-19 data/nytimes/covid-19-data
#do
#  cd $dir
#  echo "$dir"
#  echo ""
#  git pull origin master
#  cd $here
#done
#
