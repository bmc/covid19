#!/usr/bin/env bash

here=`pwd`
for dir in data/johns-hopkins/COVID-19 data/nytimes/covid-19-data
do
  cd $dir
  echo "$dir"
  echo ""
  git pull origin master
  cd $here
done
