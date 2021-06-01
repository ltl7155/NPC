for more pretrained weight of models, please download them from drive.google.com 

# download weight
`
 python data/download.py --download_weight
`

this script will download each models' weight and store them into "./data/trained_models", with a specifical name.


# download dataset
`
 python data/download.py --download_datasets
`

this script will download each models' weight and store them into "./data/trained_models", with a specifical name.


# download api
`
 python data/download.py --file_id [gogole_drive_id] --save_name [filename if None save as file_id ]
`

this script will download any google drive_id into folder with filename expect 
