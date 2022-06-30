from tkinter.constants import DISABLED, HORIZONTAL, NORMAL
from PIL import Image, ImageTk # numpy to GUI element
import tkinter as tk
from tkinter import ttk
import threading
import torch.multiprocessing as mp
import numpy as np
import athesim_functions as asf
import time
from pycromanager import Bridge # camera control
import torch

## ML-SIM App
class ML_App:
    def __init__(self,master):

        self.master = master
        master.title('Live ML-SIM')
        master.geometry("700x640") # size of gui
        tabControl = ttk.Notebook(self.master)
        
        self.tab1 = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text ='Acquisition control')
        tabControl.add(self.tab2, text ='Hardware properties')
        tabControl.place(x = 5,y = 60, width = 690, height = 575)

        self.stop_signal = mp.Queue()
        self.output = mp.Queue()
        self.stack = mp.Queue()

        # Use optosplit
        self.opto = tk.IntVar()
        self.multi = tk.Checkbutton(self.tab2,variable=self.opto)
        self.multi.place(x=15, y=120)
        # Acquire Red channel
        self.R = tk.IntVar()
        self.rChan = tk.Checkbutton(self.tab2,variable=self.R)
        self.rChan.place(x=118, y=278)
        # Acquire Green channel
        self.G = tk.IntVar()
        self.gChan = tk.Checkbutton(self.tab2,variable=self.G)
        self.gChan.place(x=118, y=301)
        # Acquire Blue channel
        self.B = tk.IntVar()
        self.bChan = tk.Checkbutton(self.tab2,variable=self.B)
        self.bChan.place(x=118, y=324)




        self.multi_label = tk.Label(self.tab2, text = "Use optosplit")
        self.multi_label.place(x = 30,y = 122)

        self.y1 = tk.IntVar()
        self.y1.set(700)
        self.yco1 = tk.Entry(self.tab2,text='Y1',textvariable=self.y1) # Y1 field
        self.yco1.place(x=75, y=280, width=25)
        self.y1_label = tk.Label(self.tab2, text = "y1")
        self.y1_label.place(x = 100,y = 280)

        self.x1 = tk.IntVar()
        self.x1.set(22)
        self.xco1 = tk.Entry(self.tab2,textvariable=self.x1) # X1 field
        self.xco1.place(x=15, y=280, width=25)
        self.x1_label = tk.Label(self.tab2, text = "x1")
        self.x1_label.place(x = 40,y = 280)        

        self.y2 = tk.IntVar()
        self.y2.set(700)
        self.yco2 = tk.Entry(self.tab2,textvariable=self.y2) # Y2 field
        self.yco2.place(x=75, y=303, width=25)
        self.y2_label = tk.Label(self.tab2, text = "y2")
        self.y2_label.place(x = 100,y=303)

        self.x2 = tk.IntVar()
        self.x2.set(695)
        self.xco2 = tk.Entry(self.tab2,textvariable=self.x2) # X2 field
        self.xco2.place(x=15, y=303, width=25)
        self.x2_label = tk.Label(self.tab2, text = "x2")
        self.x2_label.place(x = 40,y=303)  

        self.y3 = tk.IntVar()
        self.y3.set(700)
        self.yco3 = tk.Entry(self.tab2,textvariable=self.y3) # Y3 field
        self.yco3.place(x=75, y=326, width=25)
        self.y3_label = tk.Label(self.tab2, text = "y3")
        self.y3_label.place(x = 100,y=326)  

        self.x3 = tk.IntVar()
        self.x3 .set(1385)
        self.xco3 = tk.Entry(self.tab2,textvariable=self.x3) # X3 field
        self.xco3.place(x=15, y=326, width=25)
        self.x3_label = tk.Label(self.tab2, text = "x3 ")
        self.x3_label.place(x = 40,y=326) 




        self.opto_text = tk.Label(self.tab2, text = "Optosplit parameters")
        self.opto_text.place(x = 15,y=257)         

        
        self.live = tk.Button(self.tab1, width=10, text='Preview', command = self.start_live)
        self.live.place(x=15, y=20)
        self.Stop_live = tk.Button(self.tab1, width=10, text='Stop', command = self.stop_live)
        self.Stop_live.place(x=15, y=50)

        blank = np.zeros((512,512))
        blank = blank.astype('uint8')
        img =  ImageTk.PhotoImage(image=Image.fromarray(blank)) # image
        self.panel = tk.Label(self.tab1, image=img)
        self.panel.configure(image=img) # update the GUI element
        self.panel.image = img  
        self.panel.place(x=150, y=20)

        imgo = Image.open('optosplit.jpg')
        test =  ImageTk.PhotoImage(imgo)
        self.optosplit = tk.Label(self.tab2, image=test)
        self.optosplit.image = test  
        self.optosplit.place(x=150, y=20)

        imgo = Image.open('Clipboard.png')
        test =  ImageTk.PhotoImage(imgo)
        self.logo = tk.Label(image=test)
        self.logo.image = test  
        self.logo.place(x=3, y=3)


        self.quit_button = tk.Button(self.tab1,width=10, fg = "red", text='Quit',command=self.quit_gui) # quit the GUI
        self.quit_button.place(x=15, y=190)       

        self.start_live_decon = tk.Button(self.tab1,width=10, text='Live ML-SIM', command = self.start_ml_sim) # start live sim
        self.start_live_decon.place(x=15, y=80)

        self.update_ROI = tk.Button(self.tab2,width=10, text='Update ROI') #update camera ROI
        self.update_ROI.place(x=15, y=220)

        self.expTime = tk.IntVar()
        self.expTime.set(80)
        self.exposure = tk.Entry(self.tab1,textvariable=self.expTime) # exposure time field
        self.exposure.place(x=15, y=140, width=50)

        self.iMin = tk.IntVar()
        self.iMin.set(00)
        self.limLow = tk.Entry(self.tab1,textvariable=self.iMin) # Display range field
        self.limLow.place(x=15, y=363, width=30)

        self.iMax = tk.IntVar()
        self.iMax.set(100)
        self.limHigh = tk.Entry(self.tab1,textvariable=self.iMax) # Display range field
        self.limHigh.place(x=50, y=363, width=30)

        self.rMin = tk.IntVar()
        self.rMin.set(50)
        self.rlimLow = tk.Entry(self.tab1,textvariable=self.rMin) # Reconstruction range field
        self.rlimLow.place(x=15, y=423, width=30)

        self.rMax = tk.IntVar()
        self.rMax.set(1000)
        self.rlimHigh = tk.Entry(self.tab1,textvariable=self.rMax) # Reconstruction range field
        self.rlimHigh.place(x=50, y=423, width=30)

        self.display_label = tk.Label(self.tab1, text = "Display range")
        self.display_label.place(x = 13,y = 340)
        self.display_label = tk.Label(self.tab1, text = "Reconstruction range")
        self.display_label.place(x = 13,y = 400)
        self.exposure_label = tk.Label(self.tab1, text = "Exposure time (ms)")
        self.exposure_label.place(x = 13,y = 117)

        self.xOff = tk.IntVar()
        self.xOff.set(710)
        self.xoffset = tk.Entry(self.tab2,textvariable=self.xOff) # ROI input
        self.xoffset.place(x=20, y=174, width=50)
        self.xoffset_label = tk.Label(self.tab2, text = "ROI offset")
        self.xoffset_label.place(x = 15,y = 154)

        self.yOff = tk.IntVar()
        self.yOff.set(700)
        self.yoffset = tk.Entry(self.tab2,textvariable=self.yOff) # ROI input
        self.yoffset.place(x=20, y=195, width=50)
        
        if not torch.cuda.is_available():
            self.start_live_decon['state'] = DISABLED
            print('A valid GPU is required for live ML-SIM')
        else:
            gpu_dev = torch.cuda.get_device_name(0)
            print('Using device:')
            print(gpu_dev)
    
    ## Class functions
    def update_roi(self):
        xOffset = self.xOff.get() # get ROI variables from the GUI input
        yOffset = self.yOff.get()
        if xOffset < 1500 and yOffset < 1500: # make sure ROI is valid
            self.stop_live()
            time.sleep(0.1) #wait for other processes to stop
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                ROI = [xOffset, yOffset, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI    
        
    def start_live(self):
        self.start_live_decon["state"] == DISABLED
        self.quit_button["state"] == DISABLED
        optosplit = self.opto.get()
        R = self.R.get(); G = self.G.get(); B = self.B.get()
        if optosplit == 1:
            x1 = self.x1.get() # get ROI variables from the GUI input
            y1 = self.y1.get()
            x2 = self.x2.get() # get ROI variables from the GUI input
            y2 = self.y2.get()
            x3 = self.x3.get() # get ROI variables from the GUI input
            y3 = self.y3.get()
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                if core.is_sequence_running():
                    core.stop_sequence_acquisition() # stop the camera
                xmin = min(x1,x2,x3)
                xmax = max(x1,x2,x3)
                x1 = x1-xmin; x2 = x2-xmin; x3 = x3-xmin
                width = xmax-xmin+513
                ymin = min(y1,y2,y3)
                ymax = max(y1,y2,y3)
                y1 = y1-ymin; y2 = y2-ymin; y3 = y3-ymin               
                height = ymax-ymin+513
                ROI = [xmin, ymin, width, height] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI for optosplit')
        else:
            with Bridge() as bridge: # load camera control library
                x1 = self.xOff.get() # get ROI variables from the GUI input
                y1 = self.yOff.get()
                x2 = 0
                y2 = 0
                x3 = 0
                y3 = 0                
                core = bridge.get_core()
                ROI = [x1, y1, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI') 


        exposure_time = self.expTime.get()
        child_max = mp.Value('d',1000)
        child_min = mp.Value('d',1)
        rchild_max = mp.Value('d',1000)
        rchild_min = mp.Value('d',1)
        self.live_process = mp.Process(target= asf.live_view, args = (self.stop_signal,
            self.output,exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))

        self.live_process.start()
        self.plotting_process = threading.Thread(target= self.plot, args = (child_max,child_min,rchild_max,rchild_min))
        self.plotting_process.start()

    def start_ml_sim(self):
        self.live["state"] == DISABLED
        self.quit_button["state"] == DISABLED
        optosplit = self.opto.get()
        R = self.R.get(); G = self.G.get(); B = self.B.get()
        if optosplit == 1:
            x1 = self.x1.get() # get ROI variables from the GUI input
            y1 = self.y1.get()
            x2 = self.x2.get() # get ROI variables from the GUI input
            y2 = self.y2.get()
            x3 = self.x3.get() # get ROI variables from the GUI input
            y3 = self.y3.get()
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                if core.is_sequence_running():
                    core.stop_sequence_acquisition() # stop the camera
                xmin = min(x1,x2,x3)
                xmax = max(x1,x2,x3)
                x1 = x1-xmin; x2 = x2-xmin; x3 = x3-xmin
                width = xmax-xmin+513
                ymin = min(y1,y2,y3)
                ymax = max(y1,y2,y3)
                y1 = y1-ymin; y2 = y2-ymin; y3 = y3-ymin
                height = ymax-ymin+513
                ROI = [xmin, ymin, width, height] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI for optosplit')
        else:
            with Bridge() as bridge: # load camera control library
                x1 = self.xOff.get() # get ROI variables from the GUI input
                y1 = self.yOff.get()
                x2 = 0
                y2 = 0
                x3 = 0
                y3 = 0
                core = bridge.get_core()
                ROI = [x1, y1, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI')

        exposure_time = self.expTime.get()
        child_max = mp.Value('d',1000)
        child_min = mp.Value('d',1)
        rchild_max = mp.Value('d',1000)
        rchild_min = mp.Value('d',1)

        self.live_process = mp.Process(target= asf.live_ml_sim, args = (self.stack,self.stop_signal,self.output,
            exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
        self.live_process.start()

        self.plotting_process = threading.Thread(target= self.plot, args = (child_max,child_min,rchild_max,rchild_min))
        self.plotting_process.start()    

    def quit_gui(self):
        self.stop_signal.put(False)
        time.sleep(1)
        self.master.destroy()

    def stop_live(self):
        self.stop_signal.put(False)
        self.start_live_decon["state"] == NORMAL
        self.live["state"] == NORMAL
        self.quit_button["state"] == NORMAL

    def plot(self,child_max,child_min,rchild_max,rchild_min):
        while True: 

            rMin = self.rMin.get()
            rchild_min.value = rMin
            rMax = self.rMax.get()
            rchild_max.value = rMax

            iMax = self.iMax.get()
            child_max.value = iMax
            iMin = self.iMin.get()
            child_min.value = iMin

            if not self.output.empty():
                image_array = self.output.get() # empty data from reconstruction pool
                if isinstance(image_array, bool):
                    print('finished acquisition')
                    break
                elif len(image_array.shape)==2:
                    # run the update function
                    image_array = image_array-np.amin(image_array)
                    image_array = image_array*(255/np.amax(image_array)) 
                    image_array = image_array.astype('uint8')
                    img =  ImageTk.PhotoImage(image=Image.fromarray(image_array,mode='L')) # convert numpy array to tikner object 
                    self.panel.configure(image=img) # update the GUI element
                    self.panel.image = img  
                elif len(image_array.shape)==3:
                    r = image_array[:,:,0]
                    g = image_array[:,:,1]
                    result = np.zeros((512,512,3))
                    r = r-np.amin(r)
                    r = 255*(r/np.amax(r))
                    g = g-np.amin(g)
                    g = 255*(g/np.amax(g))
                    result[:,:,0] = r
                    result[:,:,1] = g
                    
                    result = result.astype('uint8')
                    img =  ImageTk.PhotoImage(image=Image.fromarray(result,mode='RGB')) # convert numpy array to tikner object 
                    self.panel.configure(image=img) # update the GUI element
                    self.panel.image = img 
            # else:
                # print('imArray was empty')

if __name__ == '__main__':
    root = tk.Tk()
    my_gui = ML_App(root)
    root.mainloop()

