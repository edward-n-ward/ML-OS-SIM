### Graphical User Interface for Machine Learning Optical Sectioning ###
# Please email rm994@cam.ac.uk with any questions

## Import packages

import tkinter as tk
from tkinter import TRUE, ttk, simpledialog
from tkinter import filedialog as fd
from tkinter.constants import DISABLED, HORIZONTAL, NORMAL
from tkinter import messagebox as mb
from tkinter.messagebox import askyesno, showinfo
from tkinter.tix import *

import datetime
from PIL import Image, ImageTk, ImageEnhance, ImageFilter  # numpy to GUI element
import threading
import torch.multiprocessing as mp
import numpy as np
import OS_functions as asf # *changed
import time
import torch
import os
import math
import json
from pycromanager import Core  # camera control
import serial
import cv2
import tifffile

## Set directory

path = "" # Enter directory here
os.chdir(path)

## Set stage parameters

asi_stage_serialport = 'COM3'
asi_stage_baudrate = 115200

chatty = False # Set to true to output print statements stating what the code is doing/which functions are being called

## ML-SIM App
class ML_App:
    def __init__(self, master):

        self.master = master
        master.title('Live OS-SIM')
        master.geometry("1040x780")
        tabControl = ttk.Notebook(self.master)

        # Load Classes for the Lasers
        self.hardware = asf.OS_SIM_Lasers() #*added

        # Load data processing class
        self.stop_signal = mp.Value('i', True) #False = stop live
        self.save_signal = mp.Value('i', False)
        self.snap_signal = mp.Value('i', False)
        self.rotate = mp.Value('i', 0)
        self.img_out_q = mp.Queue()
        self.rchild_max = mp.Value('d', 1000)
        self.rchild_min = mp.Value('d', 1)
        self.save_integer_no_stacks = mp.Value('i', 0)
        self.total_no_stacks = mp.Value('i', 0)
        self.video_interval = mp.Value('i', 0)

        self.OSSIMImgPro = asf.liveOS_SIM(self.stop_signal,self.rotate,self.save_signal,self.snap_signal,self.img_out_q,self.rchild_max,
                                          self.rchild_min,self.save_integer_no_stacks,self.total_no_stacks,self.video_interval)

        # LAG Logo
        logo_img = Image.open(os.getcwd() + "\\Images for the GUI\\Clipboard.png")
        logo_img = logo_img.resize((1, 1))
        logo_img =  ImageTk.PhotoImage(logo_img)
        self.logo = tk.Label(image=logo_img)
        self.logo.image = logo_img
        self.logo.place(x=1, y=1)

        # Create tabs
        self.tab1 = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        self.tab3 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text='Acquisition control')
        tabControl.add(self.tab2, text='Hardware properties')
        tabControl.add(self.tab3, text='System parameters')
        tabControl.place(in_=self.logo, bordermode="outside", anchor="nw", relx=0, rely=1.0, x=5, y=5, width=1020, height=750)
        self.output = mp.Queue()
        self.stack = mp.Queue()

        self.auto_reconstruction = False
        self.single_colour = "NA"

        ### TAB 1 ###

        start_stop_frame = tk.LabelFrame(self.tab1, text = "Start/Stop Live Imaging", width=175, height=90) # bg="grey",
        start_stop_frame.place(x=10, y=10) 

        # Live button
        self.live_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\live_button.png")
        self.live_button_img = self.live_button_img.resize((50, 50))
        self.live_button_img = ImageTk.PhotoImage(self.live_button_img)
        self.live_button = tk.Button(start_stop_frame, image=self.live_button_img, bd=0, command=self.start_live)
        self.live_button.place(x=15, y=10)

        # Stop button
        self.stop_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\stop_button.png")
        self.stop_button_img = self.stop_button_img.resize((50, 50))
        self.stop_button_img = ImageTk.PhotoImage(self.stop_button_img)

        self.stop_button = tk.Button(start_stop_frame, image=self.stop_button_img , bd=0, command=self.stop_live) #**changed 29.01.23
        self.stop_button.place(x=90, y=10)
        self.stop_button['state'] = tk.DISABLED

        # Saving Frame / Save Frame
        saving_frame = tk.LabelFrame(self.tab1, text = "Saving", width=175, height=150) 
        saving_frame.place(in_=start_stop_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5, relwidth=1.0)

        # Start saving button
        self.start_save_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\start_save_button.jpg")
        self.start_save_button_img = self.start_save_button_img.resize((60, 40))
        self.start_save_button_img = ImageTk.PhotoImage(self.start_save_button_img)

        self.start_save_button = tk.Button(saving_frame, image=self.start_save_button_img, bd=0, command=self.start_saving)
        self.start_save_button.place(x=5, y=5)
        self.start_save_button['state'] = tk.DISABLED

        # Stop saving button
        self.stop_save_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\stop_save_button.jpg")
        self.stop_save_button_img = self.stop_save_button_img.resize((60, 40))
        self.stop_save_button_img = ImageTk.PhotoImage(self.stop_save_button_img)

        # Snap button
        self.snap_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\snap_button.jpg")
        self.snap_button_img = self.snap_button_img.resize((60, 40))
        self.snap_button_img = ImageTk.PhotoImage(self.snap_button_img)

        self.snap_button = tk.Button(saving_frame, image=self.snap_button_img, bd=0, command=self.snap_image)
        self.snap_button.place(x=80, y=5)
        self.snap_button['state'] = tk.DISABLED

        # Save video button
        self.save_video_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\save_video_button.png")
        self.save_video_button_img = self.save_video_button_img.resize((60, 40))
        self.save_video_button_img = ImageTk.PhotoImage(self.save_video_button_img)

        self.save_video_button = tk.Button(saving_frame, image=self.save_video_button_img, bd=0, command=self.save_video)
        self.save_video_button.place(x=5, y=50)
        self.save_video_button['state'] = tk.DISABLED

        # Save stack button
        self.save_stack_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\save_stack_button.png")
        self.save_stack_button_img = self.save_stack_button_img.resize((60, 40))
        self.save_stack_button_img = ImageTk.PhotoImage(self.save_stack_button_img)

        self.save_stack_button = tk.Button(saving_frame, image=self.save_stack_button_img, bd=0, command=self.save_stack)
        self.save_stack_button.place(x=80, y=50)
        self.save_stack_button['state'] = tk.DISABLED
        
        # Choose save folder button
        def select_folder():
            string = "" + datetime.datetime.now().strftime('%Y_%m_%d') # Enter starting directory here
            self.foldername = fd.askdirectory(title='Choose a folder', initialdir=string)
            showinfo(title='Selected Folder:', message=self.foldername)
            if self.foldername:
                self.OSSIMImgPro.setsaveDir(self.foldername)
            else:
                print('No folder selected, use default')

        self.select_folder_button = ttk.Button(saving_frame, text='Folder', command=select_folder)
        self.select_folder_button.place(x=5, y=100)

        # Enter save filename button
        def select_file():
            self.filename = simpledialog.askstring("Input", "Enter filename (date and time will be appended automatically)")
            showinfo(title='Filename:', message=self.filename)
            self.OSSIMImgPro.setsaveName(self.filename)

        self.select_file_button = ttk.Button(saving_frame, text='Filename', command=select_file)
        self.select_file_button.place(in_=self.select_folder_button, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        ## Laser control Frame 4

        laser_frame = tk.LabelFrame(self.tab1, text = "Laser Control", width=175, height=270) 
        laser_frame.place(in_=saving_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5, relwidth=1.0)

        self.laser561variable = 0
        self.laser488variable = 0
        self.laser405variable = 0
        self.laser647variable = 0  

        # 561 laser control (laser 1)

        # 561 On/Off button
        self.l561_button_grey_img = Image.open(os.getcwd() + "\\Images for the GUI\\l561_button_grey.png")
        self.l561_button_grey_img = self.l561_button_grey_img.resize((50, 50))
        self.l561_button_grey_img = ImageTk.PhotoImage(self.l561_button_grey_img)

        self.l561_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\l561_button.png")
        self.l561_button_img = self.l561_button_img.resize((50, 50))
        self.l561_button_img = ImageTk.PhotoImage(self.l561_button_img)
        self.l561_button = tk.Button(laser_frame, image=self.l561_button_grey_img, bd=0, command=self.laser561_on) # ***
        self.l561_button.place(x=10, y=5)

        # 561 power control
        self.l561_label = tk.Label(laser_frame, text="L561 Power (%)")
        self.l561_label.place(x=80, y=5)

        l561_power_range = list(range(1, 101))
        self.l561_selected_power = tk.StringVar()
        self.l561_power_cb = ttk.Combobox(laser_frame, textvariable=self.l561_selected_power)
        self.l561_power_cb['values'] = [l561_power_range[p] for p in range(0, 100)]
        self.l561_power_cb['state'] = 'readonly'
        self.l561_power_cb.place(x=115, y=35, width=40)
        self.l561_power_cb["state"] = tk.DISABLED

        def l561_power_changed(event):
            self.hardware.laserAOTF_power_control561(self.laser561variable, self.l561_selected_power.get()) 
           
        self.l561_power_cb.bind('<<ComboboxSelected>>', l561_power_changed)

        # 488 laser control (laser 2)

        # 488 On/Off button
        self.l488_button_grey_img = Image.open(os.getcwd() + "\\Images for the GUI\\l488_button_grey.png")
        self.l488_button_grey_img = self.l488_button_grey_img.resize((50, 50))
        self.l488_button_grey_img = ImageTk.PhotoImage(self.l488_button_grey_img)

        self.l488_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\l488_button.png")
        self.l488_button_img = self.l488_button_img.resize((50, 50))
        self.l488_button_img = ImageTk.PhotoImage(self.l488_button_img)
        self.l488_button = tk.Button(laser_frame, image=self.l488_button_grey_img, bd=0, command=self.laser488_on)  # ***
        self.l488_button.place(x=10, y=65)

        # 488 power control
        self.l488_label = tk.Label(laser_frame, text="L488 Power (%)")
        self.l488_label.place(x=80, y=65)

        l488_power_range = list(range(1, 101))
        self.l488_selected_power = tk.StringVar()
        self.l488_power_cb = ttk.Combobox(laser_frame, textvariable=self.l488_selected_power)
        self.l488_power_cb['values'] = [l488_power_range[p] for p in range(0, 100)]
        self.l488_power_cb['state'] = 'readonly'
        self.l488_power_cb.place(x=115, y=95, width=40)
        self.l488_power_cb["state"] = tk.DISABLED

        def l488_power_changed(event):
            self.hardware.power_control_488Laser(self.laser488variable, self.l488_selected_power.get()) 

        self.l488_power_cb.bind('<<ComboboxSelected>>', l488_power_changed)

        # 405 laser control (laser 3)

        # 405 On/Off button
        self.l405_button_grey_img = Image.open(os.getcwd() + "\\Images for the GUI\\l405_button_grey.png")
        self.l405_button_grey_img = self.l405_button_grey_img.resize((50, 50))
        self.l405_button_grey_img = ImageTk.PhotoImage(self.l405_button_grey_img)

        self.l405_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\l405_button.png")
        self.l405_button_img = self.l405_button_img.resize((50, 50))
        self.l405_button_img = ImageTk.PhotoImage(self.l405_button_img)
        self.l405_button = tk.Button(laser_frame, image=self.l405_button_grey_img, bd=0, command=self.laser405_on)  # ***
        self.l405_button.place(x=10, y=125)
        self.l405_button["state"] = tk.DISABLED # Enable if using this laser

        # 405 power control
        self.l405_label = tk.Label(laser_frame, text="L405 Power (%)")
        self.l405_label.place(x=80, y=125)

        l405_power_range = list(range(1, 101))
        self.l405_selected_power = tk.StringVar()
        self.l405_power_cb = ttk.Combobox(laser_frame, textvariable=self.l405_selected_power)
        self.l405_power_cb['values'] = [l405_power_range[p] for p in range(0, 100)]
        self.l405_power_cb['state'] = 'readonly'
        self.l405_power_cb.place(x=115, y=155, width=40)

        def l405_power_changed(event):
            self.hardware.power_control405Laser(self.laser405variable, self.l405_selected_power.get()) 

        self.l405_power_cb.bind('<<ComboboxSelected>>', l405_power_changed)
        self.l405_power_cb["state"] = tk.DISABLED # Enable if using this laser

        # 647 laser control (laser 4)

        # 647 On/Off button
        self.l647_button_grey_img = Image.open(os.getcwd() + "\\Images for the GUI\\l647_button_grey.png")
        self.l647_button_grey_img = self.l647_button_grey_img.resize((50, 50))
        self.l647_button_grey_img = ImageTk.PhotoImage(self.l647_button_grey_img)

        self.l647_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\l647_button.png")
        self.l647_button_img = self.l647_button_img.resize((50, 50))
        self.l647_button_img = ImageTk.PhotoImage(self.l647_button_img)
        self.l647_button = tk.Button(laser_frame, image=self.l647_button_grey_img, bd=0, command=self.laser647_on)  
        self.l647_button.place(x=10, y=185)

        # 647 power control
        self.l647_label = tk.Label(laser_frame, text="L647 Power (%)")
        self.l647_label.place(x=80, y=185)

        l647_power_range = list(range(1, 101))
        self.l647_selected_power = tk.StringVar()
        self.l647_power_cb = ttk.Combobox(laser_frame, textvariable=self.l647_selected_power)
        self.l647_power_cb['values'] = [l647_power_range[p] for p in range(0, 100)]
        self.l647_power_cb['state'] = 'readonly'
        self.l647_power_cb.place(x=115, y=215, width=40)
        self.l647_power_cb["state"] = tk.DISABLED

        def l647_power_changed(event):
            self.hardware.laser_power_control647(self.laser647variable, self.l647_selected_power.get()) 

        self.l647_power_cb.bind('<<ComboboxSelected>>', l647_power_changed)

        # Display range entries frame 5

        ML_frame = tk.LabelFrame(self.tab1, text = "ML Reconstruction", width=175, height=120)
        ML_frame.place(in_=laser_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5, relwidth=1.0)

        self.display_label = tk.Label(ML_frame, text="Display range")
        self.display_label.place(x=13, y=5)

        self.iMin = tk.IntVar()
        self.iMin.set(5) 
        self.limLow = tk.Entry(ML_frame, textvariable=self.iMin)  # Display range field
        self.limLow.place(x=15, y=25, width=35)

        self.iMax = tk.IntVar()
        self.iMax.set(1000)
        self.limHigh = tk.Entry(ML_frame, textvariable=self.iMax)  # Display range field
        self.limHigh.place(x=50, y=25, width=35)

        # Reconstruction range entries
        
        self.display_label = tk.Label(ML_frame, text="Reconstruction range")
        self.display_label.place(x=13, y=45)

        self.rMin = tk.IntVar()
        self.rMin.set(100) #1
        self.rlimLow = tk.Entry(ML_frame, textvariable=self.rMin)  # Reconstruction range field
        self.rlimLow.place(x=15, y=65, width=35)

        self.rMax = tk.IntVar()
        self.rMax.set(1000)
        self.rlimHigh = tk.Entry(ML_frame, textvariable=self.rMax)  # Reconstruction range field
        self.rlimHigh.place(x=50, y=65, width=35)

        # Display Frame 
        # Drop down menu to select imaging mode (combobox) 

        display_frame = tk.LabelFrame(self.tab1, text = "Display", width=630, height=680) 
        display_frame.place(in_=start_stop_frame, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        # #1 = Alignment mode, 2 = Stripes no reconstruction, 3 = Single Slice reconstruction, 4 = Full volume reconstruction
        self.im_mode_label = tk.Label(display_frame, text="Imaging mode:")
        self.im_mode_label.place(x=5, y=5)

        self.im_modes = ["2 Stripes no reconstruction", "1 Spot alignment", "2 Stripes no reconstruction", "3 Single slice reconstruction", "4 Full volume reconstruction"]
        self.im_mode_selected = tk.StringVar()
        self.im_mode_selected.set("2 Stripes no reconstruction")
        self.im_mode_menu = ttk.OptionMenu(display_frame, self.im_mode_selected, *self.im_modes, command = self.im_mode_changed)
        self.im_mode_menu['state'] = 'readonly'
        
        self.im_mode_menu.place(in_=self.im_mode_label, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=3)

        # Show reconstructions checkbox (show live images)
        self.show_recon = tk.IntVar()
        self.show_recon.set(True)
        self.show_recon_checkbox = tk.Checkbutton(display_frame, variable=self.show_recon)
        self.show_recon_checkbox.place(in_=self.im_mode_menu, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=3)
        self.show_recon_checkbox.select()  
        
        self.show_recon_label = tk.Label(display_frame, text="Show live images")
        self.show_recon_label.place(in_=self.show_recon_checkbox, bordermode="outside", anchor="nw", relx=0.85, rely=0, y = 1)

        # Display
        blank = np.zeros((600, 600))
        blank = blank.astype('uint8')
        img = ImageTk.PhotoImage(image=Image.fromarray(blank))  
        self.panel = tk.Label(display_frame, image=img)
        self.panel.configure(image=img)  
        self.panel.image = img
        self.panel.place(x=5, y=30)
        
        # Frame rate display
        self.frame_rate_label = tk.Label(display_frame, text = "Volume imaging rate: X")
        self.frame_rate_label.place(in_=self.panel, bordermode="outside", anchor="nw", relx=0, rely=1.0, y = 5)

        # Colours frame
        colours_frame = tk.LabelFrame(self.tab1, text = "Colour Channels and FOV", width=170, height=180)  
        colours_frame.place(in_=display_frame, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        self.full_FOV_var = tk.IntVar()
        self.opto = tk.IntVar() 

        roi_radiobuttons_1 = tk.IntVar()
        self.multichannel_radiobutton = tk.Radiobutton(colours_frame, text = "Multiple colours", variable = roi_radiobuttons_1, value = 1, command=self.multichannel_selected)
        self.multichannel_radiobutton.place(x=5, y=15)

        self.singlechannel_radiobutton = tk.Radiobutton(colours_frame, text = "Single colour", variable = roi_radiobuttons_1, value = 2, command=self.singlechannel_selected)
        self.singlechannel_radiobutton.place(x=5, y=35)      

        self.fullFOV_radiobutton = tk.Radiobutton(colours_frame, text = "Full FOV", variable = roi_radiobuttons_1, value = 3, command=self.full_FOV_selected)
        self.fullFOV_radiobutton.place(x=5, y=55)
        self.fullFOV_radiobutton.select()
   
        # Acquire Red channel
        self.R = tk.IntVar()
        self.rChan_checkbox = tk.Checkbutton(colours_frame, variable=self.R)
        self.rChan_checkbox.place(x=105, y=85)
        self.rChan_label = tk.Label(colours_frame, text="R")
        self.rChan_label.place(x=125, y=85)
        
        # Acquire Green channel
        self.G = tk.IntVar()
        self.gChan_checkbox = tk.Checkbutton(colours_frame, variable=self.G)
        self.gChan_checkbox.place(x=105, y=105)
        self.gChan_label = tk.Label(colours_frame, text="G")
        self.gChan_label.place(x=125, y=105)
        
        # Acquire Blue channel
        self.B = tk.IntVar()
        self.bChan_checkbox = tk.Checkbutton(colours_frame, variable=self.B)
        self.bChan_checkbox.place(x=105, y=125)
        self.bChan_label = tk.Label(colours_frame, text="B")
        self.bChan_label.place(x=125, y=125)

        self.enable_channel_checkboxes(False) # disable checkboxes until "single channel" is selected

        # Acquire only R channel
        self.rChan_radiobutton = tk.Radiobutton(colours_frame, text = "R only", value = tk.IntVar(), command=self.rChan_radiobutton_clicked)
        self.rChan_radiobutton.place(x=5, y=85)
        self.rChan_radiobutton.deselect()

        # Acquire only G channel
        self.gChan_radiobutton = tk.Radiobutton(colours_frame, text = "G only", value = tk.IntVar(), command=self.gChan_radiobutton_clicked)
        self.gChan_radiobutton.place(x=5, y=105)
        self.gChan_radiobutton.deselect()

        # Acquire only B channel
        self.bChan_radiobutton = tk.Radiobutton(colours_frame, text = "B only", value = tk.IntVar(), command=self.bChan_radiobutton_clicked)
        self.bChan_radiobutton.place(x=5, y=125)
        self.bChan_radiobutton.deselect()

        ## Imaging parameter entry frame 6 

        im_param_frame = tk.LabelFrame(self.tab1, text = "Imaging Parameters", width=170, height=200) 
        im_param_frame.place(in_=colours_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5, relwidth=1.0)

        # Exposure time entry
        self.expTime = tk.IntVar()
        self.expTime.set(50) 
        self.exposure_entry = tk.Entry(im_param_frame, textvariable=self.expTime)  # exposure time field
        self.exposure_entry.place(x=5, y=5, width=30) 

        self.exposure_label = tk.Label(im_param_frame, text="Exposure time (ms)")
        self.exposure_label.place(in_=self.exposure_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        # 3D Imaging Parameters
        self.threeD_im_params_label = tk.Label(im_param_frame, text="3D Imaging Parameters")
        self.threeD_im_params_label.place(in_=self.exposure_entry, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1)

        # Z min entry
        self.zMinVal = tk.IntVar()
        self.zMinVal.set(0)
        self.zMin_entry = tk.Entry(im_param_frame, textvariable=self.zMinVal)  # Display range field
        self.zMin_entry.place(in_=self.threeD_im_params_label, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, width = 30)

        self.zMin_label = tk.Label(im_param_frame, text="Z Min (\u03bcm)")
        self.zMin_label.place(in_=self.zMin_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        self.zMin_entry['state'] = tk.DISABLED # return-to-zero code won't work if this is non zero

        # Z max entry
        self.zMaxVal = tk.IntVar()
        self.zMaxVal.set(6)
        self.zMax_entry = tk.Entry(im_param_frame, textvariable=self.zMaxVal)  # Display range field
        self.zMax_entry.place(in_=self.zMin_entry, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, width = 30)

        self.zMax_text = tk.Label(im_param_frame, text="Z Max (\u03bcm)")
        self.zMax_text.place(in_=self.zMax_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        # No. of steps entry
        self.zStepsVal = tk.IntVar()
        self.zStepsVal.set(7)
        self.zSteps_entry = tk.Entry(im_param_frame, textvariable=self.zStepsVal)  # Display range field
        self.zSteps_entry.place(in_=self.zMax_entry, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, width = 30)

        self.zSteps_text = tk.Label(im_param_frame, text="No. of steps")
        self.zSteps_text.place(in_=self.zSteps_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        self.zSteps_entry["state"] = tk.DISABLED
        self.zMin_entry["state"] = tk.DISABLED
        self.zMax_entry["state"] = tk.DISABLED

        # Video parameters
        self.vid_params_text = tk.Label(im_param_frame, text="Video parameters")
        self.vid_params_text.place(in_=self.zSteps_entry, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1)

        # No. of timepoints entry
        self.noTimePts = tk.IntVar()
        self.noTimePts.set(10)
        self.noTimePts_entry = tk.Entry(im_param_frame, textvariable=self.noTimePts)  # Display range field
        self.noTimePts_entry.place(in_=self.vid_params_text, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, width = 30)

        self.noTimePts_text = tk.Label(im_param_frame, text="No. of time points")
        self.noTimePts_text.place(in_=self.noTimePts_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        # Time intervals entry
        self.TimeInt = tk.IntVar()
        self.TimeInt.set(60)
        self.TimeInt_entry = tk.Entry(im_param_frame, textvariable=self.TimeInt)  # Display range field
        self.TimeInt_entry.place(in_=self.noTimePts_entry, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, width = 30)

        self.TimeInt_text = tk.Label(im_param_frame, text="Time Interval (s)")
        self.TimeInt_text.place(in_=self.TimeInt_entry, bordermode="outside", anchor="nw", relx=1.0, rely=0, x=5)

        self.enable_3D_im_params(False)  # When GUI starts it's in im mode 2, so disable options to select 3D imaging parameters
        self.enable_video_im_params(False)

        # Quit button (no frame)
        self.quit_button_img = Image.open(os.getcwd() + "\\Images for the GUI\\quit_button.jpg")
        self.quit_button_img = self.quit_button_img.resize((60, 35))
        self.quit_button_img = ImageTk.PhotoImage(self.quit_button_img)
        self.quit_button = tk.Button(self.tab1, image=self.quit_button_img, bd=0, command=self.quit_gui)
        self.quit_button.place(in_=ML_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5)

        # Sliders to control brightness - brightness_frame

        brightness_frame = tk.LabelFrame(self.tab1, text = "Display Brightness", width=250, height=255)
        brightness_frame.place(in_=im_param_frame, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=5, relwidth = 1.0)

        self.r_brightness = tk.DoubleVar()
        self.r_brightness.set(1)
        self.r_brightness_slider = tk.Scale(brightness_frame, from_=0, to=2, resolution = 0.1, orient=HORIZONTAL, variable=self.r_brightness)
        self.r_brightness_slider.place(x=5, y=5, width=100)
        
        self.r_brightness_slider_label = tk.Label(brightness_frame, text="R")
        self.r_brightness_slider_label.place(in_=self.r_brightness_slider, bordermode="outside", anchor="sw", relx=1.0, rely=1.0, x=5)

        self.g_brightness = tk.DoubleVar()
        self.g_brightness.set(1)
        self.g_brightness_slider = tk.Scale(brightness_frame, from_=0, to=2, resolution = 0.1, orient=HORIZONTAL, variable=self.g_brightness)
        self.g_brightness_slider.place(in_=self.r_brightness_slider, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, relwidth=1.0)
        
        self.g_brightness_slider_label = tk.Label(brightness_frame, text="G")
        self.g_brightness_slider_label.place(in_=self.g_brightness_slider, bordermode="outside", anchor="sw", relx=1.0, rely=1.0, x=5)

        self.b_brightness = tk.DoubleVar()
        self.b_brightness.set(1)
        self.b_brightness_slider = tk.Scale(brightness_frame, from_=0, to=2, resolution = 0.1, orient=HORIZONTAL, variable=self.b_brightness)
        self.b_brightness_slider.place(in_=self.g_brightness_slider, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, relwidth=1.0)
        
        self.b_brightness_slider_label = tk.Label(brightness_frame, text="B")
        self.b_brightness_slider_label.place(in_=self.b_brightness_slider, bordermode="outside", anchor="sw", relx=1.0, rely=1.0, x=5)

        self.brightness = tk.DoubleVar()
        self.brightness.set(1)
        self.brightness_slider = tk.Scale(brightness_frame, from_=0, to=2, resolution = 0.1, orient=HORIZONTAL, variable=self.brightness)
        self.brightness_slider.place(in_=self.b_brightness_slider, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=1, relwidth=1.0)
        
        self.brightness_slider_label = tk.Label(brightness_frame, text="All")
        self.brightness_slider_label.place(in_=self.brightness_slider, bordermode="outside", anchor="sw", relx=1.0, rely=1.0, x=5)
        
        # Reset button for brightness
        self.reset_Brightness_control_button = tk.Button(brightness_frame, width=10, text='Reset', command=self.reset_Brightness_control) 
        self.reset_Brightness_control_button.place(in_=self.brightness_slider, bordermode="outside", anchor="nw", relx=0, rely=1.0, y=15)

        if not torch.cuda.is_available():
            self.live_button['state'] = tk.DISABLED # disable live button 
            if chatty: print('A valid GPU is required for live OS-SIM')
        else:
            gpu_dev = torch.cuda.get_device_name(0)
            if chatty: print(['Using device:', gpu_dev])

        ### TAB 2 ###

        # ROI offset
        self.update_ROI_button = tk.Button(self.tab2, width=10, text='Update ROI', command=self.update_roi)  # update camera ROI
        self.update_ROI_button.place(x=5, y=105)

        self.xOff = tk.IntVar()
        self.xOff.set(762)
        self.xoffset = tk.Entry(self.tab2, textvariable=self.xOff)  # ROI input
        self.xoffset.place(x=10, y=59, width=50)
        self.xoffset_label = tk.Label(self.tab2, text="ROI offset")
        self.xoffset_label.place(x=5, y=39)

        self.yOff = tk.IntVar()
        self.yOff.set(614)
        self.yoffset = tk.Entry(self.tab2, textvariable=self.yOff)  # ROI input
        self.yoffset.place(x=10, y=85, width=50)

        # Optosplit parameters
        self.opto_text = tk.Label(self.tab2, text="Optosplit parameters")
        self.opto_text.place(x=15, y=257)

        redX = 104
        redY = 804
        greenX = 789
        greenY = 826
        blueX = 1486
        blueY = 819 # These values will need to be calibrated for your optosplit

        # X1 Entry
        self.x1 = tk.IntVar()
        self.x1.set(redX)
        self.xco1 = tk.Entry(self.tab2, textvariable=self.x1)  # X1 field
        self.xco1.place(x=15, y=280, width=35)
        self.x1_label = tk.Label(self.tab2, text="x1")
        self.x1_label.place(x=50, y=280)

        #Y1 Entry
        self.y1 = tk.IntVar()
        self.y1.set(redY)
        self.yco1 = tk.Entry(self.tab2, text='Y1', textvariable=self.y1)  # Y1 field
        self.yco1.place(x=75, y=280, width=35)
        self.y1_label = tk.Label(self.tab2, text="y1")
        self.y1_label.place(x=110, y=280)

        # X2 Entry
        self.x2 = tk.IntVar()
        self.x2.set(greenX)
        self.xco2 = tk.Entry(self.tab2, textvariable=self.x2)  # X2 field
        self.xco2.place(x=15, y=303, width=35)
        self.x2_label = tk.Label(self.tab2, text="x2")
        self.x2_label.place(x=50, y=303)

        # Y2 Entry
        self.y2 = tk.IntVar()
        self.y2.set(greenY)
        self.yco2 = tk.Entry(self.tab2, textvariable=self.y2)  # Y2 field
        self.yco2.place(x=75, y=303, width=35)
        self.y2_label = tk.Label(self.tab2, text="y2")
        self.y2_label.place(x=110, y=303)

        # X3 Entry
        self.x3 = tk.IntVar()
        self.x3.set(blueX)
        self.xco3 = tk.Entry(self.tab2, textvariable=self.x3)  # X3 field
        self.xco3.place(x=15, y=326, width=35)
        self.x3_label = tk.Label(self.tab2, text="x3 ")
        self.x3_label.place(x=50, y=326)

        # Y3 Entry
        self.y3 = tk.IntVar()
        self.y3.set(blueY)
        self.yco3 = tk.Entry(self.tab2, textvariable=self.y3)  # Y3 field
        self.yco3.place(x=75, y=326, width=35)
        self.y3_label = tk.Label(self.tab2, text="y3")
        self.y3_label.place(x=110, y=326)

        # Optosplit diagram
        test = ImageTk.PhotoImage(Image.open(os.getcwd() + "\\Images for the GUI\\optosplit.jpg"))
        self.optosplit_diagram = tk.Label(self.tab2, image=test)
        self.optosplit_diagram.image = test
        self.optosplit_diagram.place(x=170, y=20)

        # ROI size

        self.ROI_width = tk.IntVar()
        self.ROI_width.set(512)
        self.ROI_width_entry = tk.Entry(self.tab2, textvariable=self.ROI_width)
        self.ROI_width_entry.place(x=5, y=200, width=50)
        self.ROI_width_label = tk.Label(self.tab2, text="ROI width (<512)")
        self.ROI_width_label.place(x=60, y=200)

        self.ROI_height = tk.IntVar()
        self.ROI_height.set(512)
        self.ROI_height_entry = tk.Entry(self.tab2, textvariable=self.ROI_height)
        self.ROI_height_entry.place(x=5, y=223, width=50)
        self.ROI_height_label = tk.Label(self.tab2, text="ROI height (<512)")
        self.ROI_height_label.place(x=60, y=223)

        ### TAB 3 ###
        self.galvo_label = tk.Label(self.tab3, text="Galvo Voltages (V)")
        self.galvo_label.place(x=10, y=100)

        # Enter galvo voltage 1
        self.galvo_v1_label = tk.Label(self.tab3, text="V1 (V)")
        self.galvo_v1_label.place(x=10, y=130)

        self.galvo_v1 = tk.DoubleVar()
        self.galvo_v1.set(2.2002)
        self.galvo_v1_entry = tk.Entry(self.tab3, textvariable=self.galvo_v1)
        self.galvo_v1_entry.place(x=60, y=130, width=55)

        # Enter galvo voltage 2
        self.galvo_v2_label = tk.Label(self.tab3, text="V2 (V)")
        self.galvo_v2_label.place(x=10, y=160)

        self.galvo_v2 = tk.DoubleVar()
        self.galvo_v2.set(2.2018)
        self.galvo_v2_entry = tk.Entry(self.tab3, textvariable=self.galvo_v2)
        self.galvo_v2_entry.place(x=60, y=160, width=55)

        # Enter galvo voltage 3
        self.galvo_v3_label = tk.Label(self.tab3, text="V3 (V)")
        self.galvo_v3_label.place(x=10, y=190)

        self.galvo_v3 = tk.DoubleVar()
        self.galvo_v3.set(2.2036)
        self.galvo_v3_entry = tk.Entry(self.tab3, textvariable=self.galvo_v3)
        self.galvo_v3_entry.place(x=60, y=190, width=55)

        # Update galvo voltages button
        self.update_galvo_v_button = tk.Button(self.tab3, width=10, text='Update', command=self.update_galvo_v) 
        self.update_galvo_v_button.place(x=60, y=220)

        # Reset galvo voltages button
        self.reset_galvo_v_button = tk.Button(self.tab3, width=10, text='Reset', command=self.reset_galvo_v)  
        self.reset_galvo_v_button.place(x=60, y=250)


    ### FUNCTIONS ###

    def get561power(self):
        if self.laser561variable == True:
            power = self.l561_selected_power.get()
            try:
                power = int(power)
                return(power)
            except ValueError:
                str = "?"
                return str
        else:
            str = "Off"
            return str
        
    def get488power(self):
        if self.laser488variable == True:
            power = self.l488_selected_power.get()
            try:
                power = int(power)
                return(power)
            except ValueError:
                str = "?"
                return str
        else:
            str = "Off"
            return str

    def get647power(self):
        if self.laser647variable == True:
            power = self.l647_selected_power.get()
            try:
                power = int(power)
                return(power)
            except ValueError:
                str = "?" 
                return str
        else:
            str = "Off"
            return str

    def enable_rgb_brightness_sliders(self, value):
        if value == False:
            self.r_brightness_slider["state"] = tk.DISABLED
            self.g_brightness_slider["state"] = tk.DISABLED
            self.b_brightness_slider["state"] = tk.DISABLED
        elif value == True:
            self.r_brightness_slider["state"] = tk.NORMAL
            self.g_brightness_slider["state"] = tk.NORMAL
            self.b_brightness_slider["state"] = tk.NORMAL

    def multichannel_selected(self):
        if chatty: print('Use multiple channels radiobutton has been clicked')
        self.enable_channel_checkboxes(True)
        self.enable_channel_radiobuttons(False)
        self.reset_Brightness_control()
        self.enable_rgb_brightness_sliders(True)

        self.full_FOV_var.set(0)
        self.opto.set(1)

    def singlechannel_selected(self):
        if chatty: print('Use single channel radiobutton has been clicked')
        self.enable_channel_checkboxes(False)
        self.enable_channel_radiobuttons(True)
        self.reset_Brightness_control()
        self.enable_rgb_brightness_sliders(False)

        self.rChan_checkbox.deselect()
        self.gChan_checkbox.deselect()
        self.bChan_checkbox.deselect()

        self.gChan_radiobutton.select() # Automatically set to green to prevent no channel from being selected 
        self.gChan_radiobutton_clicked()

        self.full_FOV_var.set(0)
        self.opto.set(0)

    def full_FOV_selected(self):
        if chatty: print('Use full FOV radiobutton has been clicked')

        if self.im_mode_selected.get() == "4 Full volume reconstruction" or self.im_mode_selected.get() == "3 Single slice reconstruction":
            print("Can't image in modes 3 or 4 with full FOV")
            self.im_mode_selected.set("2 Stripes no reconstruction")
            self.im_mode_changed("2 Stripes no reconstruction")

        self.enable_channel_checkboxes(False)
        self.enable_channel_radiobuttons(False)
        self.reset_Brightness_control()
        self.enable_rgb_brightness_sliders(False)
        
        self.rChan_checkbox.deselect()
        self.gChan_checkbox.deselect()
        self.bChan_checkbox.deselect()
        
        self.xOff.set(1)
        self.yOff.set(1)
        self.full_FOV_var.set(1)
        self.opto.set(0)
        core = Core()
        ROI = [0, 0, 2048, 2048]  # build ROI 
        print('ROI values are', ROI)
        core.set_roi(*ROI)  # set ROI
      
    def rChan_radiobutton_clicked(self):
        if chatty: print('R channel only')
        self.xOff.set(self.x1.get()) 
        self.yOff.set(self.y1.get())
        self.update_roi()
        self.single_colour = "Red"

    def gChan_radiobutton_clicked(self):
        if chatty: print('G channel only')
        self.xOff.set(self.x2.get())
        self.yOff.set(self.y2.get())
        self.update_roi()
        self.single_colour = "Green"

    def bChan_radiobutton_clicked(self):
        if chatty: print('B channel only')
        self.xOff.set(self.x3.get())
        self.yOff.set(self.y1.get())
        self.full_FOV_var.set(0)
        self.update_roi()
        self.single_colour = "Blue"

    def im_mode_changed(self,value):
            if chatty: print('Imaging mode has been changed') 
            if value == "1 Spot alignment":
                self.OSSIMImgPro.set_videoMode(1)
                self.rMax.set(600000) 
                self.enable_3D_im_params(False)
                self.enable_video_im_params(False)
                if chatty: print("1 Spot alignment selected")

            elif value == "2 Stripes no reconstruction":
                self.OSSIMImgPro.set_videoMode(2)
                self.rMax.set(1000)
                self.iMin.set(1)
                self.iMax.set(1000)
                self.enable_3D_im_params(False)
                self.enable_video_im_params(False)
                if chatty: print("2 Stripes no reconstruction mode selected")

            elif value == "3 Single slice reconstruction":
                self.rMax.set(1000)
                self.iMax.set(1)
                self.iMin.set(1)
                self.OSSIMImgPro.set_videoMode(3)
                self.enable_3D_im_params(False)
                self.enable_video_im_params(True)
                if chatty: print("3 Single slice reconstruction mode selected")
                if self.full_FOV_var.get() == 1:
                    print("Can't image in modes 3 or 4 with full FOV")
                    self.full_FOV_var.set(0)
                    self.singlechannel_radiobutton.select()
                    self.rChan_radiobutton.select()
                    self.rChan_radiobutton_clicked()

            else: # Mode 4
                self.OSSIMImgPro.set_videoMode(4)
                self.rMax.set(1000)
                self.iMax.set(40000)
                self.iMin.set(1)
                self.enable_3D_im_params(True)
                self.enable_video_im_params(True)
                if chatty: print("4 Full volume reconstruction mode selected")
                if self.full_FOV_var.get() == 1:
                    print("Can't image in modes 3 or 4 with full FOV")
                    self.full_FOV_var.set(0)
                    self.singlechannel_radiobutton.select()
                    self.rChan_radiobutton.select()
                    self.rChan_radiobutton_clicked() # Also updates ROI

    ## Class functions

    def enable_3D_im_params(self, value):
        if value == False:
            self.zSteps_entry["state"] = tk.DISABLED
            self.zMin_entry["state"] = tk.DISABLED
            self.zMax_entry["state"] = tk.DISABLED
        elif value == True:
            self.zSteps_entry["state"] = tk.NORMAL
            self.zMax_entry["state"] = tk.NORMAL

    def enable_video_im_params(self, value):
        if value == False:
            self.noTimePts_entry["state"] = tk.DISABLED
            self.TimeInt_entry["state"] = tk.DISABLED
        elif value == True:
            self.noTimePts_entry["state"] = tk.NORMAL
            self.TimeInt_entry["state"] = tk.NORMAL

    def reset_Brightness_control(self):
        self.brightness.set(1)
        self.r_brightness.set(1)
        self.g_brightness.set(1) 
        self.b_brightness.set(1)

    def reset_galvo_v(self):
        self.galvo_v1.set(2.2)
        self.galvo_v2.set(2.2011)
        self.galvo_v3.set(2.203)
        self.update_galvo_v()

    def update_galvo_v(self):
        g1 = self.galvo_v1.get()
        g2 = self.galvo_v2.get()
        g3 = self.galvo_v3.get()
        new_galvo_v = np.array([g1, g2, g3]) 
        self.OSSIMImgPro.set_phaseVoltages(new_galvo_v)
        if chatty: print('Get galvo Vs: ', self.OSSIMImgPro.get_phaseVoltages())

    def update_roi(self):
        if chatty: ('Update_roi function called')
        # Get ROI variables from the GUI
        if self.xOff.get() < 1500:
            xOffset = self.xOff.get()
        else:
            xOffset = 1499
            self.xOff.set(1499)
        
        if self.yOff.get() < 1500:
            yOffset = self.yOff.get()
        else:
            yOffset = 1499
            self.yOff.set(1499)

        if self.ROI_width.get() > 2000:
            ROI_width = 512
            self.ROI_width.set(512)
        else:
            ROI_width = self.ROI_width.get()

        if self.ROI_height.get() > 2000:
            ROI_height = 512
            self.ROI_height.set(512)
        else:
            ROI_height = self.ROI_height.get()

        time.sleep(0.1)  # Wait for other processes to stop
        core = Core()
        ROI = [xOffset, yOffset, ROI_width, ROI_height]  # Build ROI
        print('ROI values are', ROI)
        core.set_roi(*ROI)  # Set ROI

    def laser561_on(self):
        if chatty: print('561 laser turned on')
        self.l561_button.configure(image=self.l561_button_img)
        self.l561_button.configure(command=self.laser561_off)
        self.laser561variable = True
        self.hardware.set561LaserState(self.laser561variable)
        self.l561_power_cb["state"] = tk.NORMAL

    def laser561_off(self):
        if chatty: print('561 laser turned off')
        self.l561_button.configure(image=self.l561_button_grey_img)
        self.l561_button.configure(command=self.laser561_on)
        self.laser561variable = False
        self.hardware.set561LaserState(self.laser561variable) 
        self.l561_power_cb["state"] = tk.DISABLED
        
    def laser488_on(self):
        if chatty: print('488 laser turned on')
        self.l488_button.configure(image=self.l488_button_img)
        self.l488_button.configure(command=self.laser488_off)
        self.laser488variable = True
        self.hardware.set488LaserState(self.laser488variable) 
        self.l488_power_cb["state"] = tk.NORMAL

    def laser488_off(self):
        if chatty: print('488 laser turned off')
        self.l488_button.configure(image=self.l488_button_grey_img)
        self.l488_button.configure(command=self.laser488_on)
        self.laser488variable = False
        self.hardware.set488LaserState(self.laser488variable) 
        self.l488_power_cb["state"] = tk.DISABLED

    def laser405_on(self):
        if chatty: print('405 laser turned on')
        self.l405_button.configure(image=self.l405_button_img)
        self.l405_button.configure(command=self.laser405_off)
        self.laser405variable = True
        self.hardware.laser_control405(self.laser405variable) 
        self.l405_power_cb["state"] = tk.NORMAL

    def laser405_off(self):
        if chatty: print('405 laser turned off')
        self.l405_button.configure(image=self.l405_button_grey_img)
        self.l405_button.configure(command=self.laser405_on)
        self.laser405variable = False
        self.hardware.laser_control405(self.laser405variable) 
        self.l405_power_cb["state"] = tk.DISABLED

    def laser647_on(self):
        if chatty: print('647 laser turned on')
        self.l647_button.configure(image=self.l647_button_img)
        self.l647_button.configure(command=self.laser647_off)
        self.laser647variable = True
        self.hardware.laser_control647(self.laser647variable) 
        self.l647_power_cb["state"] = tk.NORMAL

    def laser647_off(self):
        if chatty: print('647 laser turned off')
        self.l647_button.configure(image=self.l647_button_grey_img)
        self.l647_button.configure(command=self.laser647_on)
        self.laser647variable = False
        self.hardware.laser_control647(self.laser647variable) 
        self.l647_power_cb["state"] = tk.DISABLED

    def start_live(self):

        self.live_button["state"] = tk.DISABLED
        self.quit_button["state"] = tk.DISABLED
        self.im_mode_menu['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        self.exposure_entry['state'] = tk.DISABLED

        self.enable_channel_checkboxes(False)
        self.enable_channel_radiobuttons(False)
        self.enable_roi_radiobuttons(False)
        self.enable_3D_im_params(False)
        self.enable_video_im_params(False)

        optosplit = self.opto.get()
        print('Start_live function has been called with opto = ', optosplit)
        
        if optosplit == 0: # Don't use optosplit
            self.OSSIMImgPro.set_Optosplit(False)
            self.colourROIstr = "N/A"
            self.colourstr = "N/A"
            if self.single_colour == "NA": print("Colour error!") 
            else: self.OSSIMImgPro.set_single_colour(self.single_colour)
            self.OSSIMImgPro.set_colours([False, False, False])
            continue_var = True

            self.redVoltages = np.array([2.2001, 2.2069, 2.2088]) # These values will need to be calibrated for your system
            self.greenVoltages = np.array([2.2037, 2.2069, 2.2096])
            self.blueVoltages = np.array([2.2047, 2.2063, 2.2075])

            if self.single_colour == "Red":
                print("Imaging in red only")
                self.OSSIMImgPro.set_phaseVoltages(self.redVoltages)
            elif self.single_colour == "Green":
                print("Imaging in green only")
                self.OSSIMImgPro.set_phaseVoltages(self.greenVoltages)
            elif self.single_colour == "Blue":
                print("Imaging in blue only")
                self.OSSIMImgPro.set_phaseVoltages(self.blueVoltages)             

        elif optosplit == 1: # Use Optosplit
            self.OSSIMImgPro.set_Optosplit(True)

            R = bool(self.R.get())
            G = bool(self.G.get())
            B = bool(self.B.get())

            self.redblueVoltages = np.array([2.2022, 2.2047, 2.2075]) # red and blue
            self.redgreenVoltages = np.array([2.2037, 2.2052, 2.2069]) # green and red
            self.greenblueVoltages = np.array([2.2047, 2.2063, 2.2075]) # green and blue

            if R == True and G == True and B == False:
                self.OSSIMImgPro.set_phaseVoltages(self.redgreenVoltages)
            elif R == True and G == False and B == True:
                self.OSSIMImgPro.set_phaseVoltages(self.redblueVoltages)
            elif R == False and G == True and B == True:
                self.OSSIMImgPro.set_phaseVoltages(self.greenblueVoltages)

            if R + G + B == 0:
                if mb.askyesno("No colour selected", "You forgot to pick a colour/colours! Continue with all three colours?"):
                    self.rChan_checkbox.select()
                    self.gChan_checkbox.select()
                    self.bChan_checkbox.select()
                    R = True
                    G = True
                    B = True
                    continue_var = True
                else:
                    continue_var = False
            elif R + G + B == 1:
                continue_var = mb.askyesno("Only one colour selected", "Continue using Optosplit to image in only in one colour? (Recommend selecting Single colour option)")
            else:
                continue_var = True

            print("Continue var = ", continue_var)             

            if continue_var:
                self.colours = [R, G, B]
                self.colourstr = str(self.colours)
                self.OSSIMImgPro.set_colours(self.colours)
                if chatty: print('Colour channels RGB: ', self.OSSIMImgPro.get_colours())

                x1 = self.x1.get() # Get ROI variables from the GUI input
                y1 = self.y1.get()
                x2 = self.x2.get() 
                y2 = self.y2.get()
                x3 = self.x3.get()  
                y3 = self.y3.get()
                
                core = Core()
                if core.is_sequence_running():
                    core.stop_sequence_acquisition() # Stop the camera
                xmin = min(x1, x2, x3) 
                a = xmin % 8
                if a != 0: # Camera wants x offset and width to be multiples of 8
                    xmin = math.floor(xmin/8) * 8
                xmax = max(x1, x2, x3)
                x1_new = x1 - xmin # Relative to x1 in the new frame
                x2_new = x2 - xmin
                x3_new = x3 - xmin
                
                width = xmax - xmin + 513
                a = width % 8
                if a != 0: # Camera wants x offset and width to be multiples of 8
                    width = math.ceil(width/8) * 8
                
                ymin = min(y1, y2, y3) 
                ymax = max(y1, y2, y3)
                y1_new = y1 - ymin
                y2_new = y2 - ymin
                y3_new = y3 - ymin
                
                height = ymax - ymin + 513
                ROI = [xmin, ymin, width, height]  # Build ROI
                core.set_roi(*ROI)  # Set ROI
                if chatty: print('Successfully set ROI for optosplit')
                print('ROI set as', ROI) 
                self.colourROIvalues = np.array([x1_new, y1_new, x2_new, y2_new, x3_new, y3_new])
                self.colourROIstr = np.array2string(self.colourROIvalues)
                self.OSSIMImgPro.set_colourROI(self.colourROIvalues) 

        #####################
        if continue_var:
            if self.im_mode_selected.get() == "1 Spot alignment": val = 1
            elif self.im_mode_selected.get() == "2 Stripes no reconstruction": val = 2
            elif self.im_mode_selected.get() == "3 Single slice reconstruction": val = 3
            elif self.im_mode_selected.get() == "4 Full volume reconstruction": 
                val = 4
                StartZ = self.zMinVal.get()
                StopZ = self.zMaxVal.get()
                noSteps = self.zStepsVal.get()
                noTimePts = self.noTimePts.get()
                vid_interval = self.TimeInt.get()
                
                self.OSSIMImgPro.setStartZ(StartZ)
                self.OSSIMImgPro.setStopZ(StopZ)
                self.OSSIMImgPro.setnoSteps(noSteps)
                self.OSSIMImgPro.setnoTimePts(noTimePts)
                self.OSSIMImgPro.setvid_interval(vid_interval)

            else: val = 0

            self.enable_save_buttons(val)

            exposure_time = self.expTime.get()
            self.OSSIMImgPro.set_exposure(exposure_time)

            self.stop_signal.value = True
            self.live_process = mp.Process(target= self.OSSIMImgPro.live_start)
            self.live_process.start()

            self.plotting_process = threading.Thread(target=self.plot)
            self.plotting_process.start()
                    
        if not continue_var: # Undo disabling buttons for live mode
            self.quit_button["state"] = tk.NORMAL
            self.stop_button["state"] = tk.DISABLED
            self.live_button["state"] = tk.NORMAL
            self.enable_save_buttons(False)
            self.enable_roi_radiobuttons(True) 
            self.exposure_entry['state'] = tk.NORMAL
            self.im_mode_menu['state'] = tk.NORMAL 

            value = self.im_mode_selected.get()
            if value == "1 Spot alignment":
                self.enable_3D_im_params(False)
                self.enable_video_im_params(False)
            elif value == "2 Stripes no reconstruction":
                self.enable_3D_im_params(False)
                self.enable_video_im_params(False)
            elif value == "3 Single slice reconstruction":
                self.enable_3D_im_params(False)
                self.enable_video_im_params(True)
            else: # Mode 4
                self.enable_3D_im_params(True)
                self.enable_video_im_params(True)

    def quit_gui(self):
        print("quit_gui function called")      
        time.sleep(1)

        # Turn off all lasers
        self.laser647_off()
        self.laser561_off()
        self.laser488_off()
        # self.laser405_off()

        quit_value = 1
        if quit_value:      
            self.stop_signal.value = False 
            time.sleep(1)
            self.master.destroy()
        
        print("quit_gui function complete")

    def stop_live(self):
        self.stop_saving()
        self.stop_signal.value = False 
        self.quit_button["state"] = tk.NORMAL

        self.stop_button["state"] = tk.DISABLED
        self.live_button["state"] = tk.NORMAL
        self.enable_save_buttons(False)
        self.enable_roi_radiobuttons(True) 
        self.exposure_entry['state'] = tk.NORMAL

        self.im_mode_menu['state'] = tk.NORMAL 

        value = self.im_mode_selected.get()
        if value == "1 Spot alignment":
            self.enable_3D_im_params(False)
            self.enable_video_im_params(False)
        elif value == "2 Stripes no reconstruction":
            self.enable_3D_im_params(False)
            self.enable_video_im_params(False)
        elif value == "3 Single slice reconstruction":
            self.enable_3D_im_params(False)
            self.enable_video_im_params(True)
        else: # Mode 4
            self.enable_3D_im_params(True)
            self.enable_video_im_params(True)

        print('stop_live function has been called')

    def enable_save_buttons(self, value): # 0 = disable all, 1-4 = relevant imaging mode
        if value == 0 or value == 1 or value == 2:
            self.start_save_button['state'] = tk.DISABLED
            self.save_video_button['state'] = tk.DISABLED
            self.save_stack_button['state'] = tk.DISABLED
            self.snap_button['state'] = tk.DISABLED
        elif value == 3:
            self.start_save_button['state'] = tk.NORMAL
            self.save_video_button['state'] = tk.NORMAL
            self.save_stack_button['state'] = tk.DISABLED
            self.snap_button['state'] = tk.NORMAL
        elif value == 4:
            self.start_save_button['state'] = tk.NORMAL
            self.save_video_button['state'] = tk.NORMAL
            self.save_stack_button['state'] = tk.NORMAL
            self.snap_button['state'] = tk.DISABLED
        else:
            print("Save button error")
            self.start_save_button['state'] = tk.DISABLED
            self.save_video_button['state'] = tk.DISABLED
            self.save_stack_button['state'] = tk.DISABLED
            self.snap_button['state'] = tk.DISABLED

    def enable_channel_checkboxes(self, value):
        if value == True:
            self.rChan_checkbox["state"] = tk.NORMAL
            self.gChan_checkbox["state"] = tk.NORMAL
            self.bChan_checkbox["state"] = tk.NORMAL
        elif value == False:
            self.rChan_checkbox["state"] = tk.DISABLED
            self.gChan_checkbox["state"] = tk.DISABLED
            self.bChan_checkbox["state"] = tk.DISABLED
    
    def enable_channel_radiobuttons(self, value):
        if value == True:
            self.rChan_radiobutton["state"] = tk.NORMAL
            self.gChan_radiobutton["state"] = tk.NORMAL
            self.bChan_radiobutton["state"] = tk.NORMAL
        elif value == False:
            self.rChan_radiobutton["state"] = tk.DISABLED
            self.gChan_radiobutton["state"] = tk.DISABLED
            self.bChan_radiobutton["state"] = tk.DISABLED

    def enable_roi_radiobuttons(self, value):
        if value == True:
            self.multichannel_radiobutton["state"] = tk.NORMAL
            self.singlechannel_radiobutton["state"] = tk.NORMAL
            self.fullFOV_radiobutton["state"] = tk.NORMAL
        elif value == False:
            self.multichannel_radiobutton["state"] = tk.DISABLED
            self.singlechannel_radiobutton["state"] = tk.DISABLED
            self.fullFOV_radiobutton["state"] = tk.DISABLED

    def start_saving(self): 
        print('Start Saving Images')

        self.total_no_stacks.value = 0 # Continually take images until stop button is pressed
        self.video_interval.value = 0

        if self.im_mode_selected.get() == "4 Full volume reconstruction":
            self.save_integer_no_stacks.value = 1
            if chatty: print("self.save_integer_no_stacks.value in GUI code = ", self.save_integer_no_stacks.value )
        else:
            self.save_integer_no_stacks.value = 0
        
        self.save_signal.value = True
        
        self.start_save_button.configure(image = self.stop_save_button_img)
        self.start_save_button.configure(command = self.stop_saving)
        self.start_save_button['state'] = tk.NORMAL
        
        self.snap_button['state'] = tk.DISABLED
        self.save_video_button['state'] = tk.DISABLED
        self.save_stack_button['state'] = tk.DISABLED
        self.exposure_entry['state'] = tk.DISABLED
        self.zMax_entry['state'] = tk.DISABLED #
        self.zSteps_entry['state'] = tk.DISABLED
        self.select_file_button['state'] = tk.DISABLED
    
        self.select_folder_button['state'] = tk.DISABLED

    def stop_saving(self): 
        print('Stop Saving Images')
        self.save_signal.value = False

        self.start_save_button.configure(image = self.start_save_button_img)
        self.start_save_button.configure(command = self.start_saving)
        
        self.snap_button['state'] = tk.NORMAL
        self.save_video_button['state'] = tk.NORMAL
        self.save_stack_button['state'] = tk.NORMAL
        self.exposure_entry['state'] = tk.NORMAL
        self.zMax_entry['state'] = tk.NORMAL 
        self.zSteps_entry['state'] = tk.NORMAL
        self.select_file_button['state'] = tk.NORMAL
        self.select_folder_button['state'] = tk.NORMAL

    def snap_image(self): 
        self.total_no_stacks.value = 1
        self.video_interval.value = 0
        print('Start Saving Snapshot')
        self.save_integer_no_stacks.value = 0
        print("self.save_integer_no_stacks.value in GUI code = ", self.save_integer_no_stacks.value )
        self.save_signal.value = True

    def save_stack(self): 
        print("Save stack function has been called")
        self.total_no_stacks.value = 1
        self.video_interval.value = 0
        self.save_integer_no_stacks.value = 1
        print("self.save_integer_no_stacks.value in GUI code = ", self.save_integer_no_stacks.value )      
        self.save_signal.value = True

    def save_video(self): 
        print("Save video function has been called")
        if not mb.askyesno('Save Video', 'Are required lasers turned on to correct power and all colour channels and imaging parameters correct?'):
            pass
        else:
            self.total_no_stacks.value = self.noTimePts.get()
            self.video_interval.value = self.TimeInt.get() 
            print("total number of stacks = ", self.total_no_stacks.value, " video interval = ", self.video_interval.value)

            print('Start Saving Video')

            if self.im_mode_selected.get() == "4 Full volume reconstruction":
                self.save_integer_no_stacks.value = 1 
                print("self.save_integer_no_stacks.value in GUI code = ", self.save_integer_no_stacks.value)

            else:
                self.save_integer_no_stacks.value = 0 
            
            self.save_signal.value = True

    def Brightness_control(self, image_array): # Function to adjust brightness according to slider values
        brightness = self.brightness.get()
        r_brightness = self.r_brightness.get()
        g_brightness = self.g_brightness.get()
        b_brightness = self.b_brightness.get()
        image = image_array #Get image
            
        # Adjust brightness/sharpness/contrast of whole image or relevant channel

        if brightness != 1: # Alter whole image
            image = ImageEnhance.Brightness(image)
            image = image.enhance(brightness)

        elif r_brightness != 1 or g_brightness != 1 or b_brightness != 1:
            image = image.convert('RGB') # Split image into RGB channels
            r, g, b = image.split()

            if r_brightness != 1 and bool(self.R.get()): # Alter red channel
                r = ImageEnhance.Brightness(r)
                r = r.enhance(r_brightness)

            if g_brightness != 1 and bool(self.G.get()): # Alter blue channel
                g = ImageEnhance.Brightness(g)
                g = g.enhance(g_brightness)
            
            if b_brightness != 1 and bool(self.B.get()): # Alter green channel 
                b = ImageEnhance.Brightness(b)
                b = b.enhance(b_brightness)
            
            image = Image.merge('RGB', (r, g, b)) # Recombine image

        return image
    
    def save_metadata(self): 

        meta_data = {'Exposure/ms': int(self.expTime.get()),
            '561 laser power': self.get561power(),
            '488 laser power': self.get488power(),
            '647 laser power': self.get647power(),
            'Optosplit True/False': self.opto.get(), 
            'Colour ROI': self.colourROIstr,
            'Optosplit Colours': self.colourstr,
            'Single imaging colour': self.single_colour }
    
        if self.im_mode_selected.get() == "4 Full volume reconstruction":
            extra_meta_data = {'Z Min/um': int(self.zMinVal.get()), 
                'ZMax/um': int(self.zMaxVal.get()), 
                'No. of steps': int(self.zStepsVal.get()) }
            meta_data.update(extra_meta_data)

        string2 = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '\\' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'metadata.txt')

        with open(string2,'w') as file:
            file.write(json.dumps(meta_data, indent=4))

    def plot(self):
        save_metadata_var = True
        t2 = time.time()
        blank = np.zeros((512, 512))
        blank = blank.astype('uint8')
        blank_img = ImageTk.PhotoImage(image=Image.fromarray(blank))  
        string = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '/' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'reconstruction.tif')
        append_var = False
        while True: 
            self.rchild_min.value = self.rMin.get() # Reconstruction range values
            self.rchild_max.value = self.rMax.get()
            iMax = self.iMax.get() # Display range values
            iMin = self.iMin.get()            
            
            if not self.img_out_q.empty() and self.show_recon.get() == True: 
                t1 = time.time() 
                image_array = self.img_out_q.get()  # Empty data from reconstruction pool
                if isinstance(image_array, bool):
                    print('Finished acquisition (in plot function)')
                    break
                elif len(image_array.shape) == 2:
                    image_array = image_array - iMin 
                    image_array[image_array > iMax] = iMax 
                    image_array = image_array * (255 / iMax)
                    image_array = image_array.astype('uint8')
                    sf = 600/(np.max(image_array.shape)) # Get scale factor to resize array so longest dimension is 600 pixels (to fit screen)
                    img = Image.fromarray(image_array, mode='L')
                    img = img.resize((int(sf*image_array.shape[0]),int(sf*image_array.shape[1]))) # Resize image according to scale factor
                    img = self.Brightness_control(img) # Control image brightness using PIL functionalities
                    img = ImageTk.PhotoImage(image=img)  # Convert numpy array to tkinter object
                    display_time_local = round(time.time()-t2,4)
                    display_rate_local = round(1/display_time_local,4)
                    fr_string = "Display time = " + str(display_time_local) + "s; rate =" + str(display_rate_local) + " /s"
                    self.frame_rate_label.configure(text = fr_string)
                    self.panel.configure(image=img)  # Update the GUI element
                    self.panel.image = img
                    t2 = time.time()

                    if self.save_signal.value == True:
                        tifffile.imwrite(string, image_array.astype(np.int16), photometric='minisblack', append=append_var) 
                        append_var = True
                        if save_metadata_var:
                            self.save_metadata() 
                            save_metadata_var = False 
                        with open(timing_string, mode='a') as f: 
                            f.write(str(display_time_local) + '\n')
                            if chatty: print("Saving timing data, value: ", str(display_time_local))
                        
                    elif self.save_signal.value == False: # Create new strings to name the next metadata and timing data files, reset variable
                        # Enter relevant directories to save in:
                        string = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '/' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'reconstruction.tif')
                        timing_string = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '/' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'timing.txt')
                        append_var = False
                        save_metadata_var = True

                elif len(image_array.shape) == 3:
                    r = image_array[:, :, 0]
                    g = image_array[:, :, 1]
                    b = image_array[:, :, 2]
                    result = np.zeros((512, 512, 3))
                    r = r - np.amin(r)
                    r = 255 * (r / np.amax(r))
                    g = g - np.amin(g)
                    g = 255 * (g / np.amax(g))
                    b = b - np.amin(b)
                    b = 255 * (b / np.amax(b))
                    result[:, :, 0] = r
                    result[:, :, 1] = g
                    result[:, :, 2] = b
                    result = result.astype('uint8')
                    sf = 600/(np.max(result.shape)) # Get scale factor to resize array so longest dimension is 600 pixels (to fit screen)
                    img = Image.fromarray(result, mode = 'RGB')
                    img = self.Brightness_control(img) # Control image brightness using PIL functionalities
                    img = ImageTk.PhotoImage(image=img)  # Convert numpy array to tkinter object
                    display_time_local = round(time.time()-t2,4)
                    display_rate_local = round(1/display_time_local,4)
                    fr_string = "Display time = " + str(display_time_local) + "s; rate =" + str(display_rate_local) + " /s"
                    self.frame_rate_label.configure(text = fr_string)
                    self.panel.configure(image=img)  # Update the GUI element
                    self.panel.image = img
                    t2 = time.time()
                    
                    if self.save_signal.value == True:
                        tifffile.imwrite(string, result, photometric='rgb', append=append_var) #* eww
                        append_var = True
                        if save_metadata_var:
                            self.save_metadata() 
                            save_metadata_var = False 
                        with open(timing_string, mode='a') as f:
                            f.write(str(display_time_local) + '\n')
                            if chatty: print("Saving timing data, value: ", str(display_time_local))

                    elif self.save_signal.value == False:
                        # Enter relevant directories to save in:
                        string = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '/' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'reconstruction.tif')
                        timing_string = "" + datetime.datetime.now().strftime('%Y_%m_%d') + '/' + 'test' + datetime.datetime.now().strftime('_%Y_%m_%d_T%H.%M.%S'+'timing.txt')
                        append_var = False
                        save_metadata_var = True

                reconstruction_time = time.time()-t1
            elif not self.img_out_q.empty() and self.show_recon.get() == False: 
                image_array = self.img_out_q.get() # empty image out of the queue but don't do anything with it
                self.panel.configure(image=blank_img)  # update the GUI element
                self.panel.image = blank_img
                
        blank = np.zeros((512, 512))
        blank = blank.astype('uint8')
        img = ImageTk.PhotoImage(image=Image.fromarray(blank)) 
        self.panel.configure(image=img) 
        self.panel.image = img

        

### Initiate GUI ###
if __name__ == '__main__':
    print("Please open MicroManager if it's not already open")
    root = tk.Tk()
    my_gui = ML_App(root)
    root.mainloop()
