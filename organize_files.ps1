$proj_dir = $args[0]

# create necessary folders for model training script
New-Item -Path "$proj_dir\bin" -ItemType Directory
New-Item -Path "$proj_dir\models" -ItemType Directory
New-Item -Path "$proj_dir\results" -ItemType Directory
New-Item -Path "$proj_dir\csv" -ItemType Directory

# move images from no and yes folders to bin folder
Move-Item -Path "$proj_dir\no\*" -Destination "$proj_dir\bin"
Move-Item -Path "$proj_dir\yes\*" -Destination "$proj_dir\bin"

# move csv dataset partition files to csv folder
Move-Item -Path "$proj_dir\*.csv" -Destination "$proj_dir\csv"