python setup.py bdist_wheel -d ./for_wheel
python setup.py clean
rmdir /s /q build
rmdir /s /q wavefront_reconstruction.egg-info
timeout /t 30
