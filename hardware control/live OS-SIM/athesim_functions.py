import nidaqmx # microscope control
import numpy as np
import time
import torch.multiprocessing as mp
from pycromanager import Bridge
import torch
from models import *
import argparse

mp.freeze_support() 

## Grt ML-SIM params
def GetParams():
  opt = argparse.Namespace()

  # data
  opt.weights = '0216_SIMRec_0214_rndAll_rcan_continued.pth' 
  
  # input/output layer options
  opt.task = 'simin_gtout'
  opt.scale = 1
  opt.nch_in = 9
  opt.nch_out = 1

  # architecture options 
  opt.model='rcan'#'model to use'  
  opt.narch = 0
  opt.n_resgroups = 3
  opt.n_resblocks = 10
  opt.n_feats = 48
  opt.reduction = 16
    
  return opt

## Convenience function
def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

## Load ML-SIM model
def load_model():
    print('geting network params')
    opt = GetParams()
    print('building network')
    net = GetModel(opt)
    print('loading checkpoint')
    checkpoint = torch.load(opt.weights,map_location=torch.device('cuda'))
    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    net.module.load_state_dict(state_dict)
    return net

## ML-SIM reconstruction
def ml_reconstruction(stack,output,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    net = load_model()
    while True:
        with torch.no_grad():
            if not stack.empty():
                pixels = stack.get()
                if isinstance(pixels, bool):
                        break
                else:
                    pixels = pixels.astype(float)
                    iMax = rchild_max.value 
                    iMin = rchild_min.value
                    # run the reconstruction function 
                    if opto == 1:
                        result = np.zeros([512,512,3])
                        if R ==1:
                            data = torch.from_numpy(pixels)
                            data = data.cuda()
                            temp = data[y1:y1+512,x1:x1+512,:]
                            temp = torch.swapaxes(temp,0,2)
                            temp = temp.unsqueeze(0)
                            temp = temp.type(torch.FloatTensor)
                            temp = torch.clamp(temp,iMin,iMax)
                            temp = temp - torch.amin(temp)
                            temp = temp/torch.amax(temp)
                            sr = net(temp.cuda())
                            sr = torch.clamp(sr,0,1)
                            srframe = sr.cpu()
                            srframe = srframe.numpy()
                            result[:,:,0] = np.squeeze(srframe)
                        if G ==1:
                            temp = data[y2:y2+512,x2:x2+512,:]
                            temp = torch.swapaxes(temp,0,2)
                            temp = temp.unsqueeze(0)
                            temp = temp.type(torch.FloatTensor)
                            temp = torch.clamp(temp,iMin,iMax)
                            temp = temp - torch.amin(temp)
                            temp = temp/torch.amax(temp)
                            sr = net(temp.cuda())
                            sr = torch.clamp(sr,0,1)
                            srframe = sr.cpu()
                            srframe = srframe.numpy()
                            result[:,:,1] = np.squeeze(srframe)
                        if B ==1:
                            temp = data[y3:y3+512,x3:x3+512,:]
                            temp = torch.swapaxes(temp,0,2)
                            temp = temp.unsqueeze(0)
                            temp = temp.type(torch.FloatTensor)
                            temp = torch.clamp(temp,iMin,iMax)
                            temp = temp - torch.amin(temp)
                            temp = temp/torch.amax(temp)
                            sr = net(temp.cuda())
                            sr = torch.clamp(sr,0,1)
                            srframe = sr.cpu()
                            srframe = srframe.numpy()
                            result[:,:,2] = np.squeeze(srframe)
                                                                        
                        output.put(result)    

                    else:                    
                        data = torch.from_numpy(pixels)
                        data = torch.swapaxes(data,0,2)
                        data = data.unsqueeze(0)
                        data = data.type(torch.FloatTensor)
                        data = data.cuda()
                        data = torch.clamp(data,iMin,iMax)
                        data = data - torch.amin(data)
                        data = data/torch.amax(data)
                        sr = net(data.cuda())
                        sr = torch.clamp(sr,0,1)
                        sr = torch.squeeze(sr)
                        sr = torch.swapaxes(sr,0,1)
                        sr = sr.cpu()
                        
                        sr_frame = sr.numpy()
                        sr_frame = np.squeeze(sr_frame) 
                        output.put(sr_frame)
    
    output.put(False)

## Live view
def live_loop(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    print('starting acquisition')

    stop_signal.put(True)
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.08,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
        core = bridge.get_core()

        if core.is_sequence_running():
            core.stop_sequence_acquisition() # stop the camera
            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python        

        while True:
            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if output.empty():
                    for i in range(9):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(waits[i])     

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        pixels = pixels.astype('float64')
                        if opto == 1:
                            merged = np.zeros([512,512,3])
                            if R ==1:
                                q = pixels[y1:y1+512,x1:x1+512]
                                q = q/np.amax(q)
                                merged[:,:,0] = q
                                
                            if G ==1:    
                                q = pixels[y2:y2+512,x2:x2+512]
                                q = q/np.amax(q)
                                merged[:,:,1] = q

                            if B ==1:
                                q = pixels[y3:y3+512,x3:x3+512]
                                q = q/np.amax(q)
                                merged[:,:,2] = q
                            output.put(merged)
                        else:      
                            iMax = rchild_max.value 
                            iMin = rchild_min.value
                            pixels = np.clip(pixels, iMin,iMax)
                            output.put(pixels)
                   
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        output.put(False)

## Live ML-SIM
def acquisition_loop(stop_signal,stack,exposure):
    print('starting acquisition')
    stop_signal.put(True)
    
    with nidaqmx.Task() as VoltageTask, nidaqmx.Task() as CameraTask, Bridge() as bridge: # sorts out camera and microscope control
        voltages = np.array([0.95, 0.9508, 0.9516, 2.25, 2.2025, 2.205, 3.45, 3.454, 3.4548]) # microscope control values
        waits = np.array([0.08,0.008,0.008,0.06,0.008,0.008,0.06,0.008,0.008])
        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
        core = bridge.get_core()

        if core.is_sequence_running():
            core.stop_sequence_acquisition() # stop the camera
            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python        
        while True:

            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if stack.empty():
                    for i in range(9):
                        VoltageTask.write(voltages[i]) # move microscope
                        time.sleep(waits[i])     

                        CameraTask.write(True) # start acquisition
                        time.sleep(exposure/1000)
                        CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)                
                        result = core.get_last_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data
                        if i == 0:
                            merged = np.zeros([pixels.shape[0],pixels.shape[1],9])
                            merged[:,:,0] = pixels
                        else:
                            merged[:,:,i] = pixels

                    stack.put(merged)
                    
        core.stop_sequence_acquisition() # stop the camera
        CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        stack.put(False)

## Start live View
def live_view(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=live_loop, args=(stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_live)

    processes.reverse()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

## Start live ML-SIM
def live_ml_sim(stack,stop_signal,output,exposure,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B):
    processes = [] # initialise processes 
    proc_live = mp.Process(target=acquisition_loop, args=(stop_signal,stack,exposure))
    processes.append(proc_live)
    proc_recon = mp.Process(target=ml_reconstruction, args=(stack,output,opto,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
    processes.append(proc_recon)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()        
   
   
   
