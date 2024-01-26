### Functions for Machine Learning Optical Sectioning ###
# Please email rm994@cam.ac.uk with any questions

import numpy as np
import time
import torch.multiprocessing as mp
import threading
from pycromanager import Core
from pycromanager import Acquisition, multi_d_acquisition_events, start_headless
from models import *
import serial
import tifffile
import nidaqmx  # microscope control
import torch
import argparse
import datetime
from tkinter.messagebox import showinfo
from ctypes import c_char_p
import os
import json
import re
import cv2

chatty_laser = False # Set to true to print statements relating to debugging laser control
chatty_stage = False # Set to true to print statements relating to debugging stage control
chatty = False # Set to true to print extra statements about which functions are being called/what the code is doing
chatty_saving = False # Set to true to print statements relating to saving images

class backup_laser_class:
    def __init__(self):
        self.lasers = OS_SIM_Lasers()
    
    def backup_set561LaserState(self,bool):
        if chatty_laser: print("Set 561 laser state in backup laser class")
        self.lasers.set561LaserState(bool)

    def backup_laser_control647(self, bool):
        if chatty_laser: print("Set 647 laser state in backup laser class")
        self.lasers.laser_control647(bool)
    
    def backup_set488LaserState(self,bool):
        if chatty_laser: print("Set 488 laser state in backup laser class")
        self.lasers.set488LaserState(bool)
    
class OS_SIM_Lasers:
    def __init__(self):
        self.l405_serialport = 'COM8'
        self.l647_serialport = 'COM5'
        self.l488_serialport = 'COM7'
        self.l561_aotf_port = 'COM9'
        self.l405_baudrate = 19200
        self.l647_baudrate = 115200
        self.l488_baudrate = 115200
        self.l561_aotf_baudrate = 19200

        self.LaserState561 = False
        self.LaserState488 = False
        self.LaserState647 = False
        
        self.LaserPort561 = nidaqmx.Task() 
        self.LaserPortName561 = "Dev1/port0/line0"

        self.LaserPort488 = nidaqmx.Task() 
        self.LaserPortName488 = "Dev1/port0/line1"

    ## 561 Laser control
    def set561LaserState(self,bool):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.LaserPortName561)
            task.write(bool)
            if chatty_laser: print("Set 561 laser state")
            self.LaserState561 = bool      
            
    def exit561Laser(self):
        self.LaserPort561.close()

    def set561laserpower(self, laser_power):
        return self.laser561power

    def laserAOTF_power_control561(self,laservariable,LaserPower):
        laserCalcPower = round(float(int(LaserPower)*22.5/100),2) # Scaling power to AOTF scale, you will need to calibrate for your system
        if chatty_laser: print('Starting AOTF Communication for power control')

        if laservariable:

            with serial.Serial(port=self.l561_aotf_port,
                        baudrate=self.l561_aotf_baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=5,
                        xonxoff=False,
                        rtscts=False,
                        dsrdtr=False,
                        write_timeout=0) as ser:
            
                laserCommand = 'L1D' + str(laserCalcPower) + 'F093.096l1O1\r'
                ser.write(str.encode(laserCommand))
                fb561 = ser.readline().decode('ascii')

                if chatty_laser: print(fb561)
    
    def set488LaserState(self,bool):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.LaserPortName488)
            task.write(bool)

    def exit488Laser(self):
        self.LaserPort488.close()

    #488 Power Control
    def power_control_488Laser(self, laservariable, laserpower):
        
        if chatty_laser: print('Starting RS-232 Communication Setup for 488 power')

        if laservariable:

            with serial.Serial(port=self.l488_serialport,
                               baudrate=self.l488_baudrate,
                               bytesize=serial.EIGHTBITS,
                               parity=serial.PARITY_NONE,
                               stopbits=serial.STOPBITS_ONE,
                               timeout=5,
                               xonxoff=False,
                               rtscts=False,
                               dsrdtr=False,
                               write_timeout=0) as ser:
            
                laserCalcPower = float(int(laserpower)/200)

                laserCommand = 'p ' + str(laserCalcPower) +'\r'

                ser.write(str.encode(laserCommand)) 
                fb488 = ser.readline().decode('ascii')
                if chatty_laser: print(fb488)

        else: print('Turn the laser on first')

    ## 405 RS-232 Control
    def laser_control405(self,laservariable):
        if chatty_laser: print('Inside 405 Laser')

        laserSer = serial.Serial(self.l405_serialport,self.l405_baudrate)
        laserSer.bytesize = serial.EIGHTBITS # Bits per byte
        laserSer.parity = serial.PARITY_NONE
        laserSer.stopbits = serial.STOPBITS_ONE
        laserSer.timeout = 5
        laserSer.xonxoff = False # Disable software flow conrol
        laserSer.rtscts = False # Disable hardware (RTS/CTS) flow control
        laserSer.dsrdtr =  False # Disable hardware (DSR/DTR) flow control
        laserSer.writeTimeout = 0 # Timeout for write

        if chatty_laser: print('Starting RS-232 Communication Setup')


        if laservariable:
            try:
                laserSer.open()
            except Exception as e:
                if chatty_laser: print('Exception: Opening serial port: '+ str(e))
        
            if laserSer.isOpen():

                laserSer.write(str.encode('L 1\r\n'))  
                if chatty_laser: print("405 On command written")
                
                try:
                    laserResponse = laserSer.readline().decode('ascii')
                    if chatty_laser:  print("response data: " +laserResponse)
                    laserSer.flush()
                except Exception as e:
                    if chatty_laser:  print('Exception: Writing to serial port: '+ str(e))
                print('405 Laser On')
            else:
                print('Connection failure')
        else:
            try:
                laserSer.open()
            except Exception as e:
                if chatty_laser: print('Exception: Opening serial port: '+ str(e))

            if laserSer.isOpen():
                laserSer.write(str.encode('L 0\r\n')) 
                if chatty_laser: print("405 Off command written")
                try:
                    laserResponse = laserSer.readline().decode('ascii')
                    if chatty_laser: print("response data: " +laserResponse)
                    laserSer.flush()
                except Exception as e:
                    if chatty_laser: print('Exception: Writing to serial port: '+ str(e))
                print('405 Laser Off')
            else:
                print('Connection failure')

    def power_control405Laser(self,laservariable, laserpower):
        laserSer = serial.Serial(self.l405_serialport,self.l405_baudrate)
        laserSer.bytesize = serial.EIGHTBITS # bits per byte
        laserSer.parity = serial.PARITY_NONE
        laserSer.stopbits = serial.STOPBITS_ONE
        laserSer.timeout = 5
        laserSer.xonxoff = False # Disable software flow conrol
        laserSer.rtscts = False # Disable hardware (RTS/CTS) flow control
        laserSer.dsrdtr =  False # Disable hardware (DSR/DTR) flow control
        laserSer.writeTimeout = 0 # Timeout for write

        if chatty_laser: print('Starting RS-232 Communication Setup for power')

        if laservariable:
            try:
                laserSer.open()
            except Exception as e:
                if chatty_laser: print('Exception: Opening serial port: '+ str(e))
        
            if laserSer.isOpen():
                laserCommand = 'P ' + str(laserpower) +'\n'
                laserSer.write(str.encode(laserCommand)) 

                if chatty_laser: print("405 Power" + laserCommand)
                try:
                    laserResponse = laserSer.readline().decode('ascii')
                    if chatty_laser: print("response data: " +laserResponse)
                    laserSer.flush()
                except Exception as e:
                    if chatty_laser: print('Exception: Writing to serial port: '+ str(e))
                print('405 Power Set')
            else:
                print('Connection failure')
        else:
            try:
                laserSer.open()
            except Exception as e:
                if chatty_laser: print('Exception: Opening serial port: '+ str(e))

            if laserSer.isOpen():
                try:
                    laserResponse = laserSer.readline().decode('ascii')
                    if chatty_laser: print("response data: " +laserResponse)

                    showinfo( 
                    title='Warning', message=f'Laser 405 is Off - Turn On Laser to change power!'
                    )
                    laserSer.flush()
                except Exception as e:
                    if chatty_laser: print('Exception: Writing to serial port: '+ str(e))
                print('405 Laser Off')
            else:
                print('Connection failure')

    ## 647 RS-232 Control
    def laser_control647(self,laservariable):
        if chatty_laser: print('Inside 647 Laser')

        with serial.Serial(port=self.l647_serialport,
                       baudrate=self.l647_baudrate,
                       bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE,
                       stopbits=serial.STOPBITS_ONE,
                       timeout=5,
                       xonxoff=False,
                       rtscts=False,
                       dsrdtr=False,
                       write_timeout=0) as ser:
            if laservariable:
            
                ser.write(str.encode('en 1\r\n'))
                fb647 = ser.readline().decode('ascii')
                if chatty_laser: print(fb647)

                ser.write(str.encode('la on\r\n'))
                fb647 = ser.readline().decode('ascii')
                if chatty_laser: print(fb647)
            else:
                ser.write(str.encode('la off\r\n'))
                fb647 = ser.readline().decode('ascii')
                if chatty_laser: print(fb647)

    #647 Power Control
    def laser_power_control647(self,laservariable, laserpower):

        if chatty_laser: print('Starting RS-232 Communication Setup for 647 power')

        if laservariable:

            with serial.Serial(port=self.l647_serialport,
                        baudrate=self.l647_baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=5,
                        xonxoff=False,
                        rtscts=False,
                        dsrdtr=False,
                        write_timeout=0) as ser:
            
                laserCalcPower = float(int(laserpower))

                ser.write(str.encode('en 1\r\n'))
                fb647 = ser.readline().decode('ascii')
                if chatty_laser: print(fb647)

                laserCommand = 'ch 1 pow ' + str(laserCalcPower) +'\r\n'
                ser.write(str.encode(laserCommand))
                fb647 = ser.readline().decode('ascii')
                if chatty_laser: print(fb647)

        else: print('Turn the laser on first')

class ASI_Zstage:
    def __init__(self,save_signal,video_interval):
        self.asi_stage_serialport = 'COM3'
        self.save_signal = save_signal
        self.asi_stage_baudrate = 115200
        self.currentPosNo = 0
        self.video_interval = video_interval
        self.t1 = time.time()
        self.counter = 0

        self.lasers2 = backup_laser_class()
        self.colours = [True, True, True]
    
    def set_colours(self,newColours):
        if chatty: print("Set colours in asi z stage class: ", newColours)
        self.colours = newColours
    
    def set_single_colour(self,newColour):
        if chatty: print("Set single colour in asi z stage class: ", newColour)
        self.single_colour = newColour
    
    def setTimer(self, timer):
        self.t1 = timer
        if chatty: print("Set timer in stage class")
    
    def setnoTimePts(self, noTimePts):
        self.noTimePts = noTimePts
        if chatty: print("No of time points set in ASI z stage class") 

    def getvideo_interval(self):
        return self.video_interval

    def resetCurrentPos(self):
        self.currentPosNo = 0

    def setStartZ(self,startZ):
        self.startZ = startZ
        if chatty_stage: print('setStartZ function in asi stage class')

    def getStartZ(self):
        return self.startZ
        if chatty_stage: print('setStartZ function in asi stage class')

    def setStopZ(self,stopZ):
        self.stopZ = stopZ
        if chatty_stage: print('setStopZ function in asi stage class')

    def getStopZ(self):
        return self.stopZ

    def setnoSteps(self,noSteps):
        self.noSteps = noSteps
        if chatty_stage: print('setnoSteps function in asi stage class')

    def getnoSteps(self):
        return self.noSteps
    
    def setZrange(self): 
        self.zRange = np.linspace(self.startZ,self.stopZ,self.noSteps)
        print('setZrange function fin. zRange is ', self.zRange)
    
    def setMovementType(self):
        if self.stopZ > 12 or self.noSteps > 16: 
            # If total z range is big enough to cause problems OR there are too many step values to preload into the buffer, 
            # then use "slow movement" code, to set position manually in each loop
            slow_movement = True
            print("Slow mode")
        else:
            slow_movement = False
            print("Regular mode")
        
        return slow_movement
   
    def turn_lasers_on_off(self, bool):
        if chatty_laser: print("Turn lasers back on after pausing")
        if self.colours[0] or self.single_colour == "Red":
            if chatty_laser: print("Switching red laser on/off", bool)
            self.lasers2.backup_laser_control647(bool)
        if self.colours[1] or self.single_colour == "Green": 
            if chatty_laser: print("Switching green laser on/off", bool)
            self.lasers2.backup_set561LaserState(bool)
        if self.colours[2] or self.single_colour == "Blue": 
            if chatty_laser: print("Switching blue laser on/off", bool)
            self.lasers2.backup_set488LaserState(bool)

    def moveToNextZPos(self, slow_movement): 
        if slow_movement:
            pos = self.zRange[self.currentPosNo]
            self.setZPos(pos)
            time.sleep(0.01)

        self.currentPosNo += 1
        
        if self.currentPosNo == self.noSteps:
            if self.video_interval.value != 0 and self.save_signal.value == True:
                time_to_image_stack = time.time() - self.t1
                pause_time = self.video_interval.value - time_to_image_stack
                if chatty: print("Pause time = ", pause_time)

                if pause_time > 0: 
                    if pause_time > 2: # Not worth turning lasers on and off if only a 2 second pause
                        self.turn_lasers_on_off(False)
                        time.sleep(pause_time - 0.5)
                        self.turn_lasers_on_off(True)
                        time.sleep(0.5) # lasers turn on 0.5 s before imaging to allow for delay in laser/shutter response
                    else:
                        time.sleep(pause_time)
                else: print("Imaging time longer than time interval")
                self.counter = self.counter + 1
                print("Counter = ", self.counter)
                print("Time taken for stack including pause = ", time.time() - self.t1)
                if self.counter > self.noTimePts:
                    print("Finished saving video")
                    self.turn_lasers_on_off(False)
                self.t1= time.time()
            else:
                self.counter = 0
                self.t1 = time.time()

            if slow_movement:
                while pos > 10: # Loop to slowly go back down to the beginning of the z stack so sample doesn't crash into objective
                    pos = pos - 10
                    self.setZPos(pos)
                    time.sleep(0.01)
                    print("slowly down")
        
            self.currentPosNo = 0 

    def connect(self):
        if chatty_stage: print('Connecting to ASI Stage Control')
        self.ASI_StageSer = serial.Serial(self.asi_stage_serialport,self.asi_stage_baudrate)
        if chatty_stage: print('Stage baudrate is ', self.asi_stage_baudrate)
        self.ASI_StageSer.bytesize = serial.EIGHTBITS # Bits per byte
        self.ASI_StageSer.parity = serial.PARITY_NONE
        self.ASI_StageSer.stopbits = serial.STOPBITS_ONE
        self.ASI_StageSer.timeout = 5
        self.ASI_StageSer.xonxoff = False # Disable software flow conrol
        self.ASI_StageSer.rtscts = False # Disable hardware (RTS/CTS) flow control
        self.ASI_StageSer.dsrdtr =  False # Disable hardware (DSR/DTR) flow control
        self.ASI_StageSer.writeTimeout = 0 # Timeout for write

        if not self.ASI_StageSer.isOpen():
            try:
                self.ASI_StageSer.open()
            except Exception as e:
                print('Exception: Opening serial port: '+ str(e))

        if chatty_stage: print('Connection to ASI Stage Control Complete')

    def preload_positions(self):
        ASI_ZCommand = 'RBMODE X=0 Y=4 \r\n' # Clears the ring buffer and setst the axis of movement to Z axis
        self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 

        ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
        if chatty_stage: print("response data: " +ASI_StageResponse)
        self.ASI_StageSer.flush()

        for pos in self.zRange:
            for _ in range(3): # Remove this line once counter is in
                ASI_ZCommand = 'load z = ' + str(pos*10) + '\r\n' # Adds position to ring buffer
                self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 
                if chatty_stage: print('loaded position: ',pos)
                ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
                if chatty_stage: print("response data: " +ASI_StageResponse)
                self.ASI_StageSer.flush()

        ASI_ZCommand = 'TTL X=1 \r\n' # Set to trigger mode
        self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 

        ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
        if chatty_stage: print("response data: " +ASI_StageResponse)
        self.ASI_StageSer.flush()

    def disconnect(self):
        try:
            ASI_ZCommand = 'TTL X=0 \r\n' # Set to trigger mode
            self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 
            ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
            print("response data: " +ASI_StageResponse)
            self.ASI_StageSer.flush()

            self.ASI_StageSer.close()
        except Exception as e:
            print('Exception: Opening serial port: '+ str(e))

    def setZPos(self,pos): # Set Z position in um
        z_axis_value = float(pos*10)
        ASI_ZCommand = 'move z=' + str(z_axis_value) +'\r\n'
        self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 
        if chatty_stage: print("ASI Z Command" + ASI_ZCommand)
        try:
            ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
            if chatty_stage: print("response data: " +ASI_StageResponse)
            self.ASI_StageSer.flush()
        except Exception as e:
            print('Exception: Writing to serial port: '+ str(e))
        if chatty_stage: print('ASI Stage Z axis moved')

    ## Z axis control testing for OS-SIM
    def z_axis_control(self,zvalue, zstatus):
        print('Starting RS-232 Communication Setup for ASI Stage Control')
        
        z_axis_value = float(int(zvalue))

        if zstatus:
            try:
                self.ASI_StageSer.open()
            except Exception as e:
                print('Exception: Opening serial port: '+ str(e))
        
            if self.ASI_StageSer.isOpen():
                ASI_ZCommand = 'move z=' + str(z_axis_value) +'\r\n'
                self.ASI_StageSer.write(str.encode(ASI_ZCommand)) 

                if chatty_stage: print("AZI Z Command" + ASI_ZCommand)
                try:
                    ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
                    print("response data: " +ASI_StageResponse)
                    self.ASI_StageSer.flush()
                except Exception as e:
                    print('Exception: Writing to serial port: '+ str(e))
                print('ASI Stage Z axis moved')
            else:
                print('Connection failure')
        else:
            try:
                self.ASI_StageSer.open()
            except Exception as e:
                print('Exception: Opening serial port: '+ str(e))

            if self.ASI_StageSer.isOpen():
                try:
                    ASI_StageResponse = self.ASI_StageSer.readline().decode('ascii')
                    print("response data: " +ASI_StageResponse)

                    showinfo( 
                    title='Warning', message=f'AZI Stage Control is Off - Turn On Device for Control!'
                    )
                    self.ASI_StageSer.flush()
                except Exception as e:
                    print('Exception: Writing to serial port: '+ str(e))
                print('ASI Stage control off')
            else:
                print('Connection failure')

class OS_SIM_Galvo:
    def __init__(self,aoPortName,phaseVoltages,minV,maxV):
        self.aoPort = nidaqmx.Task()
        self.aoPortName = aoPortName
        self.extTrigPort = '/Dev1/PFI9'
        self.phaseVoltages = phaseVoltages
        self.minV = minV
        self.maxV = maxV
    
    #### Control Functions ####

    def initialise(self):
        self.aoPort.ao_channels.add_ao_voltage_chan(self.aoPortName,'mychannel',self.minV,self.maxV) # Add analogue output port
        self.aoPort.timing.cfg_samp_clk_timing(1000,self.extTrigPort,active_edge = nidaqmx.constants.Edge.RISING, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan = 50)
        self.aoPort.out_stream.offset = 0
        self.aoPort.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.ALLOW_REGENERATION

        self.aoPort.write(self.phaseVoltages)
        self.aoPort.start()

    def setPos(self,Pos):
        self.aoPort.write(self.phaseVoltages[Pos])

    def exit(self):
        self.aoPort.close()
        
class OS_SIM_Camera:
    def __init__(self,doPortName,exposure):
        self.doPort = nidaqmx.Task()
        self.doPortName = doPortName
        self.exposure = exposure
    
    #### Control Functions ####

    def initialise(self):
        self.doPort.do_channels.add_do_chan(self.doPortName)

    def snap(self):
        self.doPort.write(True) # make sure camera has stopped by requesting a final unused image
        self.doPort.write(False)

    def exit(self):
        self.doPort.close()

class liveOS_SIM:
    def __init__(self,stop_signal,rotate,save_signal,snap_signal,img_out_q,rchild_max,rchild_min,save_integer_no_stacks,total_no_stacks,video_interval):
        self.stop_signal = stop_signal
        self.rotate = rotate 
        self.save_signal = save_signal # True = save, False = not save
        self.snap_signal = snap_signal
        self.img_out_q = img_out_q
        self.rchild_max = rchild_max
        self.rchild_min = rchild_min
        self.save_integer_no_stacks = save_integer_no_stacks
        self.total_no_stacks = total_no_stacks
        self.video_interval = video_interval

        self.rawDataq = mp.Queue()
        self.rawDataq2 = mp.Queue()      
        self.saveq = mp.Queue() 
        self.splitq = mp.Queue()
        manager = mp.Manager()
        self.full_save_dir = manager.Value(c_char_p, '')
        self.volumeSaveCounter = mp.Value('i', 0)
        self.volumeFinish = mp.Value('i', 0)

        self.pxPitch = 0.086 # um
        self.zStep = 0.5 # um
        self.noImgs = 50 # Number of images per volume
        self.Optosplit = False # False = don't use, True = use
        self.colours = [True, True, True] # [Red, Green, Blue]
        self.single_colour = "NA"
        self.colourROI = np.array([4, 5, 710, 0, 1422, 8]) # Will need to be calibrated for your system
        
        string = "" + datetime.datetime.now().strftime('%Y_%m_%d') # changed 21.04.2023
        if not os.path.exists(string):
            os.makedirs(string)            
        self.saveDir = string
        self.saveName = 'test' # Name of saved files

        self.device = torch.device("cuda:0") # Set device to GPU
        self.dtype = torch.half # Set data type to half

        self.ML_looptime = 1
        self.videoMode = 2 # 1 = Alignment mode, 2 = Stripes no reconstruction (default), 3 = Single Slice reconstruction, 4 = Full volume reconstruction

        mp.freeze_support() 

        self.aoPortName = "Dev1/ao0"
        self.phaseVoltages = np.array([2.2, 2.2, 2.2]) # Should update when start live
        
        self.minV = 0
        self.maxV = 4

        self.doPortName = "Dev1/port0/line2"
        self.exposure = 50

    #### Getters and Setters ####

    def setsaveDir(self,newsavedir):
        if not os.path.isdir(newsavedir):
            os.mkdir(newsavedir)
        self.saveDir = newsavedir
        string = 'New save directory is: ' + self.saveDir
        print(string)

    def setsaveName(self,newsavename):
        self.saveName = newsavename
        string = 'New save filename is: ' + self.saveName
        print(string)

    def setStartZ(self,startZ): 
        self.startZ = startZ
        if chatty: print('Set start z in liveOS_SIM class')

    def setStopZ(self,stopZ):
        self.stopZ = stopZ
        if chatty: print('Set stop z in liveOS_SIM class')

    def setnoSteps(self,noSteps):
        self.noSteps = noSteps
        if chatty: print('Set no steps in liveOS_SIM class')

    def setnoTimePts(self, noTimePts):
        self.noTimePts = noTimePts
        if chatty: print("No of time points set in liveOS_SIM class")

    def setvid_interval(self, vid_interval):
        self.vid_interval = vid_interval
        if chatty: print("Video Interval set in liveOS_SIM class = ", self.vid_interval)

    def get_doPortName(self):
        return self.doPortName

    def set_doPortName(self,newdoPortName):
        self.doPortName = newdoPortName

    def get_exposure(self):
        return self.exposure

    def set_exposure(self,newExposure):
        self.exposure = newExposure

    def get_aoPortName(self):
        return self.aoPortName

    def set_aoPortName(self,newaoPortName):
        self.aoPortName = newaoPortName
    
    def get_phaseVoltages(self):
        return self.phaseVoltages

    def set_phaseVoltages(self,newVoltages):
        self.phaseVoltages = newVoltages
        if chatty: print('In liveOS SIM class, set new phase voltages of', newVoltages)

    def get_minV(self):
        return self.minV

    def set_minV(self,newminV):
        self.mivV = newminV

    def get_maxV(self):
        return self.minV

    def set_maxV(self,newminV):
        self.mivV = newminV

    def get_pxPitch(self):
        return self.pxPitch

    def set_pxPitch(self, newpxPitch):
        self.pxPitch = newpxPitch

    def get_zStep(self):
        return self.zStep

    def set_zStep(self, newzStep):
        self.zStep = newzStep

    def get_noImgs(self):
        return self.noImgs 

    def set_noImgs(self,newnoimgs):
        self.noImgs = newnoimgs

    def get_Optosplit(self):
        return self.Optosplit
    
    def set_Optosplit(self,boolVal):
        self.Optosplit = boolVal
        if chatty: print('Optosplit set to ',self.Optosplit,' in functions')

    def get_colours(self):
        return self.colours

    def set_colours(self,newColours):
        self.colours = newColours

    def set_single_colour(self,newColour):
        self.single_colour = newColour

    def get_colourROI(self):
        return self.colourROI

    def set_colourROI(self,newcolourROI):
        self.colourROI = newcolourROI
        if chatty: print("Updating colour ROI in liveOS SIM class")

    def get_videoMode(self):
        return self.videoMode

    def set_videoMode(self, mode):
        self.videoMode = mode

    #### ML Functions ####

    ## Get OS-SIM params
    def GetParamsOS(self):
        optOS = argparse.Namespace()
        optOS.weights = os.getcwd() + "\\light_model.pth"

        # input/output layer options
        optOS.task = 'simin_gtout'
        optOS.scale = 1
        optOS.nch_in = 3
        optOS.nch_out = 1
        optOS.kernel = 5 

        # architecture options 
        optOS.model='rcan' # Model to use  
        optOS.narch = 0
        optOS.n_resgroups = 3
        optOS.n_resblocks = 3 
        optOS.n_feats = 60 
        optOS.reduction = 16
            
        return optOS

    ## Convenience function
    def remove_dataparallel_wrapper(self,state_dict):
        r"""Converts a DataParallel model to a normal one by removing the "module."
        wrapper in the module dictionary

        Args:
            state_dict: a torch.nn.DataParallel state dictionary
        """
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, vl in state_dict.items():
            name = k[7:] # Remove 'module.' of DataParallel
            new_state_dict[name] = vl

        return new_state_dict

    ## Load OS-SIM model
    def load_model_OS(self):
        if chatty: 
            print("Loading OS-SIM model")
            print('Getting network params')
        optOS = self.GetParamsOS()
        if chatty: print('Building OS network')
        netOS = GetModel(optOS) # Build empty model
        if chatty: print('Loading checkpoint')
        checkpoint = torch.load(optOS.weights, map_location=torch.device('cuda')) # Load the weights
        if type(checkpoint) is dict: # Remove junk pytorch stuff
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        netOS.module.load_state_dict(state_dict) # Put weights into model
        return netOS

    #### Acquisition and Image Processing ####

    ## Camera and galvo synchronisation
    def acquisition_loop(self): 
        print('Starting acquisition') 
        print("Galvo voltages = ", self.phaseVoltages)
        galvo = OS_SIM_Galvo(self.aoPortName,self.phaseVoltages,self.minV,self.maxV)
        galvo.initialise()
        camera = OS_SIM_Camera(self.doPortName,self.exposure)
        camera.initialise()

        core = Core()
        
        if core.is_sequence_running():
            core.stop_sequence_acquisition() # Stop the camera
            camera.snap() # Take an image on the camera
        
        core.set_exposure(self.exposure) 
        core.set_property('pco_camera','Acquiremode','External') 
        core.set_property('pco_camera','Triggermode','External')
        if chatty: print("In normal acq. loop, galvo ao port name: ",self.aoPortName,", phase voltages: ",self.phaseVoltages,", min and max V: ",self.minV,self.maxV)

        core.initialize_circular_buffer()
        core.start_continuous_sequence_acquisition(0) # Start the camera

        camera.snap() # Take an image on the camera

        while core.get_remaining_image_count() == 0: # Wait until picture is available
            time.sleep(0.001)  
        result = core.pop_next_tagged_image()

        if self.videoMode == 4: 
            stage = ASI_Zstage(self.save_signal,self.video_interval)
            stage.setnoSteps(self.noSteps)
            stage.setStartZ(self.startZ)
            stage.setStopZ(self.stopZ)
            slow_movement = stage.setMovementType()
            stage.resetCurrentPos()
            stage.setZrange()
            stage.connect()
            if slow_movement:
                stage.moveToNextZPos(slow_movement)
            else: stage.preload_positions()
            stage.set_colours(self.colours) # Turn lasers on and off when taking a video
            stage.set_single_colour(self.single_colour)
            stage.setnoTimePts(self.noTimePts)  

        while True:
            if not self.stop_signal.value:
                self.splitq.put(False)
                break
            elif self.rawDataq.empty():
                t9 = time.time()
                for i in range(3):
                    t1 = time.time()
                    camera.snap() # Take an image on the camera
                    loop_counter = 0
                    while core.get_remaining_image_count() == 0: # Wait until image is available
                        loop_counter += 1
                        
                    t5 = time.time()
                    result = core.pop_next_tagged_image() # Get image data into python
                    pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # Reshape image data
                    pixels = pixels.astype('float64')

                    if self.videoMode == 1 or self.videoMode == 2:
                        if self.Optosplit:
                            merged = np.zeros([512,512,3])
                            if self.colours[0]:
                                q = pixels[self.colourROI[1]:self.colourROI[1]+512,self.colourROI[0]:self.colourROI[0]+512]
                                q = q/np.amax(q) 
                                merged[:,:,0] = q
                                
                            if self.colours[1]:    
                                q = pixels[self.colourROI[3]:self.colourROI[3]+512,self.colourROI[2]:self.colourROI[2]+512]
                                q = q/np.amax(q)
                                merged[:,:,1] = q

                            if self.colours[2]:
                                q = pixels[self.colourROI[5]:self.colourROI[5]+512,self.colourROI[4]:self.colourROI[4]+512]
                                q = q/np.amax(q)
                                merged[:,:,2] = q
                            self.splitq.put(merged)
                        else:
                            iMax = self.rchild_max.value 
                            iMin = self.rchild_min.value
                            pixels = np.clip(pixels,iMin,iMax)
                            self.splitq.put(pixels)
                    else: 
                        if i == 0:
                            merged = np.zeros([pixels.shape[0],pixels.shape[1],3])
                            merged[:,:,0] = pixels
                        else: 
                            merged[:,:,i] = pixels
                            if i == 2:
                                self.splitq.put(merged)
                if self.videoMode == 4:
                    t6 = time.time()
                    stage.moveToNextZPos(slow_movement)
                    if slow_movement == False and stage.currentPosNo == 0 and self.volumeSaveCounter.value == 0 and self.save_signal.value == 1:
                        self.splitq.put('start')
                        print("Start saving stack, time = ", time.time())
                        stage.setTimer(time.time())
                        
                        self.volumeSaveCounter.value = 1
                    if slow_movement == True and stage.currentPosNo == 1 and self.volumeSaveCounter.value == 0 and self.save_signal.value == 1: 
                        self.splitq.put('start')
                        print("Start saving stack, time = ", time.time())
                        stage.setTimer(time.time())
                        
                        self.volumeSaveCounter.value = 1
                    if stage.currentPosNo == 0 and self.save_signal.value == 0 and self.volumeSaveCounter.value == 1: 
                        print("self.save_integer_no_stacks.value (1) = ", self.save_integer_no_stacks.value)
                        self.save_integer_no_stacks.value = 0
                        print("self.save_integer_no_stacks.value (2) = ", self.save_integer_no_stacks.value)
                        print("Finished saving stack")
 
        camera.snap()
        if core.is_sequence_running():
            if chatty: print("Yes sequence is running :)")
            core.stop_sequence_acquisition() 
        camera.snap()
        camera.exit()
        galvo.exit()
        
        if self.videoMode == 4:
            stage.disconnect()      

        print('Finished Acquisition Loop (seq acq successfully stopped)')

    ## OS-SIM reconstruction
    def OS_reconstruction(self,output): #stack,OSoutput,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B
        netOS = self.load_model_OS()
        netOS.eval()
        netOS.to(torch.device('cuda'))
        while True:
            with torch.no_grad(): # Set to eval mode
                if not self.rawDataq.empty(): # Waits until 3 images are available
                    t2 = time.time()
                    pixels = self.rawDataq.get() # Get those 3 images
                    if isinstance(pixels, bool): # Stop if sent bool false
                        output.put(False)
                        break # Stop reconstructing
                    else:
                        pixels = pixels.astype(float)
                        iMax = self.rchild_max.value 
                        iMin = self.rchild_min.value 
                        # Run the reconstruction function 
                        if self.Optosplit:
                            result = np.zeros([512,512,3])
                            data = torch.from_numpy(pixels) # Turns pixel array into pytorch tensor
                            data = data.cuda() # Move pytorch tensor to GPU
                            if self.colours[0]: 
                                temp = data[self.colourROI[1]:self.colourROI[1]+512,self.colourROI[0]:self.colourROI[0]+512] # Crop to region of interest 
                                temp = torch.swapaxes(temp,0,2) # Move axes
                                temp = temp.unsqueeze(0) # Pytorch needs (1,9,512,512) array of images
                                temp = temp.type(torch.FloatTensor) #  Want to use float for precision
                                temp = torch.clamp(temp,iMin,iMax) # Match input to training data histogram
                                temp = temp - torch.amin(temp) # Normalise between 1 and 0
                                temp = temp/torch.amax(temp) # Normalise between 1 and 0
                                t7 = time.time()
                                sr = netOS(temp.cuda()) # Move temp array to GPU then perform reconstruction 
                                sr = torch.clamp(sr,0,1) # Normalise network output
                                srframe = sr.cpu() # Move output to cpu
                                srframe = srframe.numpy() # Convert back to numpy array
                                result[:,:,0] = np.squeeze(srframe) # Add reconstruction to stack of reconstructions (one per colour)
                            if self.colours[1]:
                                temp = data[self.colourROI[3]:self.colourROI[3]+512,self.colourROI[2]:self.colourROI[2]+512]
                                temp = torch.swapaxes(temp,0,2)
                                temp = temp.unsqueeze(0)
                                temp = temp.type(torch.FloatTensor)
                                temp = torch.clamp(temp,iMin,iMax)
                                temp = temp - torch.amin(temp)
                                temp = temp/torch.amax(temp)
                                t7 = time.time()
                                sr = netOS(temp.cuda())
                                sr = torch.clamp(sr,0,1)
                                srframe = sr.cpu()
                                srframe = srframe.numpy()
                                result[:,:,1] = np.squeeze(srframe)
                            if self.colours[2]:
                                temp = data[self.colourROI[5]:self.colourROI[5]+512,self.colourROI[4]:self.colourROI[4]+512]
                                temp = torch.swapaxes(temp,0,2)
                                temp = temp.unsqueeze(0)
                                temp = temp.type(torch.FloatTensor)
                                temp = torch.clamp(temp,iMin,iMax)
                                temp = temp - torch.amin(temp)
                                temp = temp/torch.amax(temp)
                                t7 = time.time()
                                sr = netOS(temp.cuda())
                                sr = torch.clamp(sr,0,1)
                                srframe = sr.cpu()
                                srframe = srframe.numpy()
                                result[:,:,2] = np.squeeze(srframe)
                                                                            
                            output.put(result) # Sends reconstructions to plotting function    

                        else: 
                            
                            data = torch.from_numpy(pixels)
                            data = data.cuda()     
                            data = torch.swapaxes(data,0,2)
                            data = data.unsqueeze(0)
                            data = data.type(torch.FloatTensor)
                            data = torch.clamp(data,iMin,iMax)
                            data = data - torch.amin(data)
                            data = data/torch.amax(data)
                            t7 = time.time()
                            sr = netOS(data)
                            sr = torch.clamp(sr,0,1)
                            sr = torch.squeeze(sr)
                            sr = torch.swapaxes(sr,0,1)
                            sr = sr.cpu()
                            
                            sr_frame = sr.numpy()
                            sr_frame = np.squeeze(sr_frame) 
                            output.put(sr_frame)

    ## Fourier Transform
    def gpu_Fourier(self):
        while True:
            if not self.rawDataq.empty(): # Waits until images are available
                pixels = self.rawDataq.get() # Get those 3 images
                if isinstance(pixels, bool): # Stop if sent bool false
                    self.img_out_q.put(False)
                    break # Stop reconstructing
                else:
                    pixels = pixels.astype(float)
                    data = torch.from_numpy(pixels) # Turns pixel array into pytorch tensor
                    data = data.cuda() # Move pytorch tensor to GPU

                    img_fft2 = torch.fft.fft2(data) # Compute 2D fourier transform
                    img_fft2 = torch.fft.fftshift(img_fft2)
                    img_fft2 = torch.abs(img_fft2)

                    iMax = self.rchild_max.value 
                    iMin = self.rchild_min.value

                    img_fft2 = torch.log(img_fft2+1)
                    img_fft2 = img_fft2 - torch.amin(img_fft2)
                    img_fft2 = img_fft2/torch.amax(img_fft2)

                    img_fft2_frame = img_fft2.cpu() # move output to cpu
                    img_fft2_frame = img_fft2_frame.numpy() # convert back to numpy array
                    img_fft2_frame = np.squeeze(img_fft2_frame) 

                    self.img_out_q.put(img_fft2_frame)
        print('Finished F.Loop')

    ## Rotating image volume
    def project_volume(self): #stop_signal,stack,rotateVal,no_imgs,height,width,voxel_ratio
        currentSlice = 0 # Variable to keep track of which slice in the volume is being processed
        
        while True:
            if not self.rawDataq2.empty(): # Waits until reconstructions are available
                slice = self.rawDataq2.get()
        
                if isinstance(slice, bool): # Stop if sent bool false
                    self.img_out_q.put(False)
                    break # Stop reconstructing
                else:

                    if self.Optosplit:
                        if currentSlice == 0:
                                result = np.zeros([512,512,3])
                                shape = np.shape(slice[:,:,0])
                                height = shape[0]
                                width = shape[1]
                                Rfinal_im = torch.zeros([height, width], device=self.device, dtype=self.dtype) 
                                Rcombined_im = torch.zeros([height, width, 2], device=self.device, dtype=self.dtype)
                                Gfinal_im = torch.zeros([height, width], device=self.device, dtype=self.dtype) 
                                Gcombined_im = torch.zeros([height, width, 2], device=self.device, dtype=self.dtype)
                                Bfinal_im = torch.zeros([height, width], device=self.device, dtype=self.dtype) 
                                Bcombined_im = torch.zeros([height, width, 2], device=self.device, dtype=self.dtype)

                        if self.colours[0]:                                          
                            image_array = slice[:,:,0]
                            image_array = torch.from_numpy(image_array.astype("float16")).to(self.device)

                            startH = 0
                            stopH = startH+height
                            Rprevimg = Rfinal_im[startH:stopH, 0:width]
                            Rfinal_im[startH:stopH, 0:width] = self.GPU_max_proj(Rcombined_im, Rprevimg, image_array)

                        if self.colours[1]:           
                            image_array = slice[:,:,1]
                            image_array = torch.from_numpy(image_array.astype("float16")).to(self.device)

                            startH = 0
                            stopH = startH+height
                            Gprevimg = Gfinal_im[startH:stopH, 0:width]
                            Gfinal_im[startH:stopH, 0:width] = self.GPU_max_proj(Gcombined_im, Gprevimg, image_array)

                        if self.colours[2]:           
                            image_array = slice[:,:,2]
                            image_array = torch.from_numpy(image_array.astype("float16")).to(self.device)

                            startH = 0
                            stopH = startH+height
                            Bprevimg = Bfinal_im[startH:stopH, 0:width]
                            Bfinal_im[startH:stopH, 0:width] = self.GPU_max_proj(Bcombined_im, Bprevimg, image_array)

                        currentSlice += 1

                        if currentSlice == self.noSteps:
                            currentSlice = 0 
                            Rdisp_array = Rfinal_im.cpu().detach().numpy()   
                            Gdisp_array = Gfinal_im.cpu().detach().numpy()   
                            Bdisp_array = Bfinal_im.cpu().detach().numpy()   
                            result[:,:,0] = Rdisp_array*(2**16)
                            result[:,:,1] = Gdisp_array*(2**16)
                            result[:,:,2] = Bdisp_array*(2**16)
                            self.img_out_q.put(result.astype("uint16")) # Full volume max projection of all the slice reconstructions
                    
                    else: # if self.optosplit = False
                        if currentSlice == 0:
                            shape = np.shape(slice)
                            height = shape[0]
                            width = shape[1]
                            final_im = torch.zeros([height, width], device=self.device, dtype=self.dtype) 
                            combined_im = torch.zeros([height, width,2], device=self.device, dtype=self.dtype)

                        image_array = torch.from_numpy(slice.astype("float16")).to(self.device)
                        startH = 0
                        stopH = startH+height
                        previmg = final_im[startH:stopH, 0:width]
                        final_im[startH:stopH, 0:width] = self.GPU_max_proj(combined_im, previmg, image_array)

                        currentSlice += 1

                        if currentSlice == self.noSteps:
                            currentSlice = 0 
                            disp_array = final_im.cpu().detach().numpy()   
                            disp_array = disp_array*(2**16)
                            self.img_out_q.put(disp_array.astype("uint16"))               
        
        print('Finished R.Loop')

    def GPU_max_proj(self, combined_im, old_image, new_image):
        combined_im[:,:,0] = old_image
        combined_im[:,:,1] = new_image
        final_im = torch.amax(combined_im, 2)
        return final_im       

    def saveQueueSplitter(self,output):
        break_flag = True
        doVolumeSaving = False
        no_images_counter = 0
        while break_flag:
            if not self.splitq.empty(): # Check that the save queue isn't empty
                pixels = self.splitq.get()
                if isinstance(pixels, bool): # Check if the live acqusition has been finished
                    output.put(False)
                    break
                else:
                    if self.save_signal.value:
                        if self.videoMode == 4: 
                            total_images_to_save = self.noSteps * self.total_no_stacks.value

                        else: 
                            total_images_to_save = self.total_no_stacks.value

                        if chatty_saving: print("Total no of images to save = ", total_images_to_save)
                        self.full_save_dir.value = self.saveDir + '/' + self.saveName + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S') + "_time1"
                        print('Save directory is:', self.full_save_dir.value)
                        
                        if self.videoMode == 4:
                            if chatty_saving: print("Saving mode (1)")
                            
                            if isinstance(pixels, str):
                                if chatty_saving: print("Saving mode (2)")
                                doVolumeSaving = True
                                while self.splitq.empty():
                                    time.sleep(1/1000)
                                pixels = self.splitq.get()
                            
                            output.put(pixels)

                            if doVolumeSaving:
                                self.saveq.put(pixels)
                                no_images_counter += 1
                                if chatty_saving: print("No of images saved so far (Saving mode 1) = ", no_images_counter)
                                if no_images_counter == total_images_to_save:
                                    if chatty_saving: print("Finished (Saving mode 1)")
                                    self.save_signal.value = False
                                    self.video_interval.value = 0
                                   
                        else:
                            output.put(pixels)
                            self.saveq.put(pixels)
                            no_images_counter += 1
                            if chatty_saving: print("No of images saved so far (Saving mode 2) = ", no_images_counter)
                            if no_images_counter == total_images_to_save:
                                if chatty_saving: print("Finished (Saving mode 2)")
                                self.save_signal.value = False
                                self.video_interval.value = 0 
                        
                        if self.snap_signal.value:
                            self.save_signal.value = False
                            self.snap_signal.value = False 
                        
                        while self.save_signal.value or self.save_integer_no_stacks.value:
                            if not self.splitq.empty(): # Check that the save queue isn't empty
                                pixels = self.splitq.get()
                                if isinstance(pixels, bool): # Check if the live acqusition has been finished
                                    output.put(False)
                                    break_flag = False
                                    break
                                else:
                                    if self.videoMode == 4:
                                        if isinstance(pixels, str):
                                            if chatty_saving: print("Saving mode (3)")
                                            doVolumeSaving = True
                                            while self.splitq.empty():
                                                time.sleep(1/1000)
                                            pixels = self.splitq.get()
                                        
                                        output.put(pixels)

                                        if doVolumeSaving:
                                            self.saveq.put(pixels)
                                            no_images_counter += 1
                                            if chatty_saving: print("No of images saved so far (Saving mode 3) = ", no_images_counter)
                                            if no_images_counter == total_images_to_save:
                                                if chatty_saving: 
                                                    print("Finished (Saving mode 3)")
                                                    print("Imaging finished = ", time.time())
                                                self.save_signal.value = False
                                                self.video_interval.value = 0 
                                                break
                                            if int(no_images_counter) % int(self.noSteps) == 0: 
                                                if chatty_saving: print("End of stack")
                                                self.saveq.put("New stack")
                                    else:
                                        output.put(pixels)
                                        self.saveq.put(pixels)
                                        no_images_counter += 1
                                        if chatty_saving: print("No of images saved so far (Saving mode 4) = ", no_images_counter)
                                        if no_images_counter == total_images_to_save:
                                            if chatty_saving: print("Finished (Saving mode 4)")
                                            self.save_signal.value = False
                                            self.video_interval.value = 0 
                                            break
                        self.saveq.put(False)
                        self.volumeSaveCounter.value = 0 
                        doVolumeSaving = False
                        no_images_counter = 0
                    else:    
                        output.put(pixels)
        print('Save splitter queue ended')

    def saveFrame(self): 
            while True:
                if not self.stop_signal.value: # Check if the live acqusition has been finished
                    break
                else:
                    if self.save_signal.value: # Check that images should be being saved 
                        if not self.saveq.empty(): # Check that the save queue isn't empty 
                            save_stack_counter = 1
                            pixels = self.saveq.get()
                            while not isinstance(pixels, bool):
                                with tifffile.TiffWriter(self.full_save_dir.value+'.tif', append=True) as tf:
                                    tf.write(pixels[:,:,0].astype(np.int16), contiguous = True)
                                    tf.write(pixels[:,:,1].astype(np.int16), contiguous = True)
                                    tf.write(pixels[:,:,2].astype(np.int16), contiguous = True)
                                    if chatty_saving: print('Image saving with dir: ', self.full_save_dir.value)                              
                                    pixels = self.saveq.get() 

                                if isinstance(pixels, str):
                                    if pixels == "New stack":
                                        save_stack_counter += 1
                                        if save_stack_counter < 10:
                                            self.full_save_dir.value = re.sub("_time\d", "_time"+str(save_stack_counter), self.full_save_dir.value)
                                            print('New save directory is:', self.full_save_dir.value)
                                        elif save_stack_counter == 10:
                                            self.full_save_dir.value = re.sub("_time\d", "_time"+str(save_stack_counter), self.full_save_dir.value)
                                            print('New save directory is:', self.full_save_dir.value)
                                        elif save_stack_counter < 100:
                                            self.full_save_dir.value = re.sub("_time\d\d", "_time"+str(save_stack_counter), self.full_save_dir.value)
                                            print('New save directory is:', self.full_save_dir.value)
                                        pixels = self.saveq.get()
                                

    #### Starting Functions ####

    ## Start live OS-SIM 
    def live_start(self): 
        if chatty: print('Running updated version live OS sim')
        processes = [] # Initialise processes 
        threads = []

        if self.videoMode == 1: # Alignment mode
            print('For back-relfection spot alignment, significantly increase max of reconstruction range to approx 600,000. For fluorescence, approx. 40,000')
            proc_live = mp.Process(target=self.acquisition_loop)
            processes.append(proc_live)
            proc_fourier = mp.Process(target=self.gpu_Fourier)
            processes.append(proc_fourier)

            thread_spliting = threading.Thread(target=self.saveQueueSplitter, args = (self.rawDataq,))
            threads.append(thread_spliting)
            
        elif self.videoMode == 2: # Stripes no reconstruction
            proc_live = mp.Process(target=self.acquisition_loop)
            processes.append(proc_live)

            process_spliting = mp.Process(target=self.saveQueueSplitter, args = (self.img_out_q,))
            processes.append(process_spliting)

        elif self.videoMode == 3: # Single Slice reconstruction
            proc_live = mp.Process(target=self.acquisition_loop)
            processes.append(proc_live)
            proc_recon = mp.Process(target=self.OS_reconstruction, args = (self.img_out_q,))
            processes.append(proc_recon)

            proc_spliting = mp.Process(target=self.saveQueueSplitter, args = (self.rawDataq,))
            processes.append(proc_spliting)

        else: # Full volume reconstruction

            proc_live = mp.Process(target=self.acquisition_loop)
            processes.append(proc_live)
            proc_recon = mp.Process(target=self.OS_reconstruction, args = (self.rawDataq2,))
            processes.append(proc_recon)
            proc_rotate = mp.Process(target=self.project_volume)
            processes.append(proc_rotate) 

            thread_spliting = threading.Thread(target=self.saveQueueSplitter, args=(self.rawDataq,))
            threads.append(thread_spliting)             

        thread_saving = threading.Thread(target=self.saveFrame)
        threads.append(thread_saving)
        
        threads.reverse()
        for thread in threads:
            thread.start()

        processes.reverse()
        for process in processes: # Start each process
            process.start()

        print('Finished Start Function')

if __name__ == "__main__": # For testing
    stop_signal = mp.Value('i', True)
    save_signal = mp.Value('i', False)
    rotate = mp.Value('i', 10)
    img_out_q = mp.Queue()
    rchild_max = mp.Value('i', 255)
    rchild_min = mp.Value('i', 0)
    snap_signal = False

    print("This code doesn't work by itself - call this script from the OS_GUI script")

    OSSIM = liveOS_SIM(stop_signal,rotate,save_signal,snap_signal,img_out_q,rchild_max,rchild_min) # Added snap_signal

    OSSIM.set_videoMode(3)

    fps_thread = threading.Thread(target= OSSIM.live_start)
    fps_thread.start()
    fps_thread.join()





