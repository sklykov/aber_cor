:: This script assumes that the conda environmet already exist and the package their installed
:: Just for info that conda installed (?: maybe for further logic will be valuable)
conda.exe -V
:: recommended by conda call
call conda.bat activate wfs_wheel_test   
:: call python with command line
python -c "from wfs_reconstruction import gui_wfs_reconstruction; gui_wfs_reconstruction.launch()"
