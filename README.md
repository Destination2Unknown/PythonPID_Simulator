# PythonPID_Simulator
Python PID Controller and Process Simulator (FOPDT) with GUI.

Run the File.


Then select Model Values and Tune PID..
______________________________

![GUI](https://user-images.githubusercontent.com/92536730/147006723-46e4d353-c0d4-44f0-b5a8-d93925699b8e.JPG)



Hit Refresh to show trends
______________________

![PID_Trends](https://user-images.githubusercontent.com/92536730/147006704-422bcf11-6ae4-4b0b-9399-59a71ba094e9.JPG)


Requires:
pip -> cmd python get-pip.py
numpy -> cmd pip install numpy
matplotlib -> cmd pip install matplotlib
scipy -> cmd pip install scipy

To create Exe use windows cmd:
cmd -> pip install pyinstaller

cmd -> cd to folder with PythonPID_Simulator.py (change py to pyw = no console window)
Then run pyinstaller with the -F flag to produce the standalone exe:
pyinstaller -F PythonPID_Simulator.py
It will output to dist/PythonPID_Simulator.exe