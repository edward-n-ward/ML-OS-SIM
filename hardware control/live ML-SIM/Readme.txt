Readme for live ML-SIM GUI
******************************************************

The following code is provided to show the use of ML-SIM for real-time acquisition.
This GUI has been tested with Python 3.7.9, CUDA 11.5, Pytorch 1.10.1, Pycromanager 0.16.4 
and nidaqmx 0.6.1, all of which can be installed through anaconda and pip.


The main funtion is AtheSIM_GUI.py which will launch the GUI. There is a library of functions
in athesim_functions.py which handles the core functionality. Also included are the .pth pytorch
files that define the ML netowrk used. For a comparison of these see [1,2]

The program makes use of the 
multiprocessing library provided by pytorch to run 3 three parallel threads:

	1) The acquisition thread "acquisition_loop" which controls the camera and DAQ card through
	the pycromanager and nidaqmx. This acquires a stack of nine images and places the image data
	into the queue variable "stack". The acquisition will not start again until the stack variable
	has been emptied.
	2) The deconvolution thread "ml_reconstruction" which takes the data from teh queue and performs
	either a single or multi-channel ML-IM reconstruction on the GPU. Once reconstructed this then 
	puts the data into the queue variable "output". The reconstruction will not start again until
	the "output" queue is empty and the "stack" queue is full.
	3) The GUI is run in its own thread which allows interactivity while the live reconstruction
	is ongoing. Withing this thread is the plotting function "plot" which takes the queue variable 
	"ouput" and displays it as either an RGB or single colour image. The stop function allows for 
	the acquisition to be terminated by forcing a string vairable into the queues. All parallel 
	functions will terminate once a string is detected in the queue.      

This GUI has ben designed to operate on a SIM system where pattern orintation and phase are controlled
by a single analogue output on a DAQ card. However, this could be changed relatively easiy by altering
only the "acquisition_loop" function. 

*********************************************************

[1] Christensen CN, Ward EN, Lu M, Lio P, Kaminski CF. "ML-SIM: universal reconstruction of structured 
illumination microscopy images using transfer learning." Biomed Opt Express. 2021 Apr 15;12(5):2720-2733.
[2] Christensen CN, Lu M, Ward EN, Lio P, Kaminski CF. "Spatio-temporal Vision Transformer for 
Super-resolution Microscopy" arXiv:2203.00030
