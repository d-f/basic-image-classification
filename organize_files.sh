#!/bin/bash 

echo "Enter project directory"
read proj_dir

# create necessary folders for model training script
mkdir "$proj_dir/bin"
mkdir "$proj_dir/models"
mkdir "$proj_dir/results"
mkdir "$proj_dir/csv"

#  move images from no and yes folders to bin folder
mv $proj_dir/no/* $proj_dir/bin
mv $proj_dir/yes/* $proj_dir/bin

# move csv dataset partition files to csv folder
mv $proj_dir/*.csv $proj_dir/csv