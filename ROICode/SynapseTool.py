
import csv

from tkinter import *
from tkinter import filedialog
from tkinter import ttk as ttk

import DataRead as DR
import DataAnalyze as DA
import GenFolderStruct as GFS
import shutil

import numpy as np

import os.path


"""========================================================================================"""

class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 1000     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
    

class MainWindow:
    def __init__(self):
        """Generate a the main selection screen of the Synapse tool"""
        root = Tk()
        root.title("Synapse Tool")

        self.folder_path = ""

        self.b1 = Button(text="Choose Directory", command=self.browse_button)
        self.b1.grid(row=0,column=1,padx=5, pady=5)

        b1_ttp = CreateToolTip(self.b1, 'Choose the directory we will consider')

        self.l1 = Label(root, text = self.folder_path,bg="light blue")
        self.l1.grid(row=1,column=1,padx=5, pady=5)

        self.bGenDat = Button(text="Read data",command = lambda:self.GenData_button())
        self.bGenDat.grid(row=2,column=1,padx=5, pady=5)

        bGen_ttp = CreateToolTip(self.bGenDat, 'Construct data from experimental data')

        self.bAnDat = Button(text="Analyze data", command=lambda:self.AnalData_button())
        self.bAnDat.grid(row=2,column=2,padx=5, pady=5)

        bAn_ttp = CreateToolTip(self.bAnDat, 'Analyze generated data')

        self.bDirStruct = Button(text="Generate folders",command = lambda:self.DirStruct_button())
        self.bDirStruct.grid(row=2,column=0,padx=5, pady=5)

        bDir_ttp = CreateToolTip(self.bDirStruct, 'Transform the data to the correct structure')

        self.bClearDat = Button(text="Clear data", command=lambda:self.CleanData_button())
        self.bClearDat.grid(row=3,column=1,padx=5, pady=5)

        bGen_ttp = CreateToolTip(self.bClearDat, 'Clear directory for a fresh restart')

        mainloop()

    def browse_button(self):
        """Function that sets the desired folder"""

        filename = filedialog.askdirectory()
        Dir =str(filename)

        self.l1.config(text="Folder: " + Dir)
        self.folder_path = str(filename)

    def GenData_button(self):
        """Function that generates the read data window"""

        root2 = Toplevel()
        app2 = GenerateWindow(root2,self.folder_path)

    def AnalData_button(self):
        """Function that generates the anaylze data window"""

        #try:
        Acceptable = []
        tdict      = {}

        for c in sorted(os.listdir(self.folder_path)):
            cellType = os.path.split(self.folder_path)[-1]

            if(os.path.exists(self.folder_path+'/'+c+'/Synapses'+cellType+'_'+c+".csv")):
                xall = [x if ((x[-3:]=='lsm') or (x[-3:]=='tif')) else '' for x in os.listdir(self.folder_path+'/'+c)]
                lenTime = len(list(filter(('').__ne__, xall)))
                Acceptable.append(c+'/')

                # Check if the cells were stimulated so that this can be flagged to the user
                SynCSV = self.folder_path+'/'+c+'/Synapses'+cellType+'_'+c+".csv"

                with open(SynCSV, newline='') as f:

                    reader = csv.reader(f)
                    row = np.array([float(x) for x in next(reader)])
                    tdict[c] = self.CheckDat(row,lenTime)

        root4 = Toplevel()
        app4 = DataAnalyzeWindow(root4,self.folder_path,Acceptable,tdict)

        #except:
        #    alert_popup("Warning","Oops - something went wrong")

    def CheckDat(self,row1,lenTime):

        """Flag that passes wether the normalized data was a success"""

        temp = row1[4:lenTime+4]-row1[-2*lenTime:-lenTime]
        temp2 = temp[3]/np.nanmean(temp[:3])

        # 1 and 1.2 are chosen relatively arbitrary and can be changed
        if(temp2<1):
            check = 2
        elif(temp2<1.2):
            check = 1
        else:
            check = 0

        return check

    def CleanData_button(self):
        """Function that generates the clean data window"""
        root3 = Toplevel()
        app3 = DeleteWindow(root3,self.folder_path)

    def DirStruct_button(self):
        """Function that generates the directory generation window"""
        root5 = Toplevel()
        app5 = DirStructWindow(root5)

"""========================================================================================"""

class DirStructWindow:
    """Class that defines the directory structure window"""
    def __init__(self,master):

        self.master = master
        self.frame = Frame(self.master)

        self.source_path = ""
        self.target_path = ""

        self.b1 = Button(master,text="Choose source directory", command=self.source_button)
        self.b1.grid(row=0,column=0,padx=5, pady=5)

        self.l1 = Label(master, text = self.source_path,bg="light blue")
        self.l1.grid(row=1,column=0,padx=5, pady=5)


        self.b2 = Button(master,text="Choose target directory", command=self.target_button)
        self.b2.grid(row=0,column=1,padx=5, pady=5)

        self.l2 = Label(master, text = self.target_path,bg="light blue")
        self.l2.grid(row=1,column=1,padx=5, pady=5)

        self.l3 = Label(master, text = 'New folder name:')
        self.l3.grid(row=2,column=0,padx=5, pady=5)

        self.e1 = Entry(master,width=10)
        self.e1.grid(row=2,column=1,padx=5, pady=5)

        self.progress = ttk.Progressbar(master, orient = HORIZONTAL, length = 100, mode = 'determinate')
        self.progress.grid(row=3,column=0,padx=5, pady=5)

        self.bgo     = Button(master,text="Go!", command=self.go_button)
        self.bgo.grid(row=3,column=1,padx=5, pady=5)

    def source_button(self):
        """ Allow user to select a directory and store it in global var called source_path """

        filename = filedialog.askdirectory()
        Dir =str(filename)
        self.l1.config(text="Folder: " + os.path.split(Dir)[-1])
        self.source_path = str(filename)

    def target_button(self):
        """ Allow user to select a directory and store it in global var called target_path """

        filename = filedialog.askdirectory()
        Dir =str(filename)
        self.l2.config(text="Folder: " + os.path.split(Dir)[-1])
        self.target_path = str(filename)

    def go_button(self):
        """ Generate the target directory, and deleting the directory if it already exists"""
        try:
            GFS.CreateCellDirs(self.source_path,self.target_path,self.e1.get(),self)
        except:
            self.deleteDir_popup()

    def deleteDir_popup(self):
        """ Window that gives users last chance to rethink if they really want to delete the files and
        send them to the nether realm"""
        root = Toplevel()
        root.title('Warning!')

        m = 'Do you want to delete ' + self.e1.get() + '? \n There might be files in it.'
        w = Label(root, text=m)
        w.grid(row=0,column=0,padx=5, pady=5)

        b1 = Button(root, text="OK", command=lambda:self.DeleteDir(root), width=10)
        b1.grid(row=1,column=0,padx=5, pady=5)

        b2 = Button(root, text="Cancel", command=root.destroy, width=10)
        b2.grid(row=1,column=1,padx=5, pady=5)

    def DeleteDir(self,root):
        """ It's done - the files will be deleted."""
        if(os.path.exists(self.target_path+'/'+self.e1.get())):
            shutil.rmtree(self.target_path+'/'+self.e1.get())
            root.destroy()

"""========================================================================================"""

class GenerateWindow(Frame):
    def __init__(self,master,Dir):

        """Class that defines the read data window"""

        self.master = master
        self.frame = Frame(self.master)

        self.Dir   = Dir
        self.v = StringVar()

        self.MC = IntVar()

        Label(master, text="Select synapse data").grid(row = 0, column = 0)

        self.Proj = StringVar(master)
        self.Proj.set('Max')
        choices2 = sorted({ "Sum","Max","Min","Mean","Median","None"})
        self.ProjMen = OptionMenu(master, self.Proj, *choices2)
        self.ProjMen.grid(row = 0, column = 2)
        Zp_ttp = CreateToolTip(self.ProjMen, 'z-projection')

        self.Mode = StringVar(master)
        self.Mode.set('Luminosity')
        choices = sorted({ "Area","Luminosity","Soma","Puncta","Dendrite"})
        self.zpMenu = OptionMenu(master, self.Mode, *choices)
        self.zpMenu.grid(row = 1, column =2)
        pZ_ttp = CreateToolTip(self.zpMenu, 'Select if you want to measure area or Luminosity')
    

        self.e1 = Entry(master,width=5)
        self.e1.grid(row=3,column=1,padx=5, pady=5)

        e1_ttp = CreateToolTip(self.e1, 'Enter the first cell you want to anaylze')

        self.e2 = Entry(master,width=5)
        self.e2.grid(row=3,column=2,padx=5, pady=5)

        self.e3 = Entry(master,width=5)
        self.e3.grid(row=1,column=1,padx=5, pady=5)
        Label(master, text="\u03BCm per pixel").grid(row = 0, column = 1)

        self.chk = Checkbutton(master, text='Multi-channel', variable=self.MC)
        self.chk.grid(row=1,column=0,padx=5, pady=5)

        e2_ttp = CreateToolTip(self.e2, 'Enter the last cell you want to analyze')

        Label(master, text="Cells").grid(row = 3, column = 0)

        self.progress = ttk.Progressbar(master, orient = HORIZONTAL, length = 100, mode = 'determinate')
        self.progress.grid(row=5,column=0,padx=5, pady=5)

        self.l1 = Label(master, textvariable=self.v,bg="white",width=40)
        self.l1.grid(row=5,column=1,padx=5, pady=5)

        self.b1 = Button(master,text="Go!",command = self.Gen_button)
        self.b1.grid(row=6,column=1,padx=5, pady=5)

        self.b2= Button(master,text="Clear",command = self.clearlab)
        self.b2.grid(row=5,column=2,padx=5, pady=5)

        b2_ttp = CreateToolTip(self.b2, 'Clear the textfield')

    def close_windows(self):
        self.master.destroy()

    def clearlab(self):
        self.v.set("")

    def Gen_button(self):
        # Allow user to select a directory and store it in global var
    # called folder_path
        #try:
        if(self.e2.get()==""):
            a = int(self.e1.get())
        else:
            a = int(self.e2.get())

        if(self.e3.get()==""):
            b = 1.0
        else:
            b = float(self.e3.get())


        if (self.Proj.get()=="None"):
            proj = None
        else:
            proj = self.Proj.get()
            
        for cell in ["cell_"+str(i) for i in range(int(self.e1.get()),a+1)]:
            DR.FullEval(self.Dir+"/"+cell+"/",self.Mode.get(),bool(self.MC.get()),b,proj,self)

        #except:
        #    alert_popup("Warning","Oops - something went wrong")

"""========================================================================================"""

class DataAnalyzeWindow(Frame):

    """Class that defines the analyze data window"""
    def __init__(self,master,Dir,Acceptable,tdict):

        self.master = master
        self.Dir    =  Dir
        self.frame = Frame(self.master)

        self.var = dict()
        self.varbool = dict()
        self.allvar = IntVar()
        choices = { 'Dynamic line','Contour plot','Buckets'}

        self.tkvar = StringVar(master)
        self.tkvar.set('Dynamic line')
        count=1

        chk = Checkbutton(master, text='All', variable=self.allvar,command=lambda:self.reset())
        chk.grid(row=0,column=0,padx=5, pady=5)

        for cell in Acceptable:

            self.var[cell[:-1]]=IntVar()
            if(tdict[cell[:-1]]==2):
                chk_color = "red"
            elif(tdict[cell[:-1]]==1):
                chk_color = "orange"
            elif(tdict[cell[:-1]]==0):
                chk_color = "black"

            chk = Checkbutton(master, text=cell[:-1], variable=self.var[cell[:-1]],fg=chk_color,
                              command=lambda key=cell[:-1]: self.Readstatus(key))
            chk.grid(row=count,column=0,padx=5, pady=5)

            count += 1

            self.varbool[cell[:-1]]=False

        self.b1 = Button(master,text="Go!",command = self.IntPlotWindow)
        self.b1.grid(row=int(count/2)+1,column=1,padx=5, pady=5)

        self.popupMenu = OptionMenu(master, self.tkvar, *choices)
        self.popupMenu.grid(row = int(count/2), column =1)

        self.e1 = Entry(master,width=5)
        self.e1.grid(row = int(count/2)-1, column =1)

    def IntPlotWindow(self):
        """Function that,based on the selection, runs the right analysis window"""

        Data,lenTime = DA.GetData(self.Dir,self.varbool)

        if(self.tkvar.get()=='Dynamic line'):
            DA.DataAnalWindow(Data,self.Dir,self.varbool,lenTime)

        elif(self.tkvar.get()=='Contour plot'):
            if(self.e1.get() == ''):
                DA.ContourWindow(Data,self.Dir,self.varbool,lenTime)
            else:
                contNum = [int(x) for x in self.e1.get().split(',')][0]
                DA.ContourWindow(Data,self.Dir,self.varbool,lenTime,contNum)

        elif(self.tkvar.get()=='Buckets'):
            Thresh = [float(x) for x in self.e1.get().split(',')]
            DA.MultiLineWindow(Data,self.Dir,Thresh,self.varbool,lenTime)

    def Readstatus(self,key):
        """Sets checkbuttons to the correct value"""

        var_obj = self.var.get(key)
        self.varbool[key] = bool(var_obj.get())

    def reset(self):
        """Resets buttons if All is clicked"""

        if self.allvar.get():

            for cell in self.var:
                var_obj = self.var.get(cell)
                var_obj.set(1)
                self.varbool[cell] = bool(var_obj.get())

        else:

            for cell in self.var:
                var_obj = self.var.get(cell)
                var_obj.set(0)
                self.varbool[cell] = bool(var_obj.get())

"""========================================================================================"""

class DeleteWindow(Frame):
    """Class that defines the clean data window"""
    def __init__(self,master,Dir):

        self.master = master
        self.frame = Frame(self.master)

        self.Dir   = Dir
        self.var1 = IntVar()
        self.var2 = IntVar()
        self.var3 = IntVar()
        self.var4 = IntVar()

        c1 = Checkbutton(master, text="Shifting Direction", variable=self.var1)
        c1.grid(row=2,column=0,padx=5, pady=5)

        c2 = Checkbutton(master, text="Dendrite Files", variable=self.var2)
        c2.grid(row=3,column=0,padx=5, pady=5)

        c3 = Checkbutton(master, text="Json Files", variable=self.var3)
        c3.grid(row=4,column=0,padx=5, pady=5)

        c4 = Checkbutton(master, text="Roi picture", variable=self.var4)
        c4.grid(row=5,column=0,padx=5, pady=5)


        Label(master, text="Minimum Cell").grid(row = 0, column = 0)
        Label(master, text="Maximum Cell").grid(row = 1, column = 0)

        self.e1 = Entry(master)
        self.e1.grid(row=0,column=1,padx=5, pady=5)

        self.e2 = Entry(master)
        self.e2.grid(row=1,column=1,padx=5, pady=5)

        self.b1 = Button(master,text="Clean",command = self.clean_button)
        self.b1.grid(row=3,column=1,padx=5, pady=5)

    def clean_button(self):
        # Based on the selection the window will delete the chosen files - giving a warning
        # if they dont exist

        #try:
        temp = os.path.split(self.Dir)[-1]

        if(self.e2.get()==""):
            a = int(self.e1.get())

        else:
            a = int(self.e2.get())

        for cell in ["cell_"+str(i) for i in range(int(self.e1.get()),a+1)]:
            for x in os.listdir(self.Dir+"/"+cell+"/"):
                if("MinDir" in x and self.var1.get()==1):
                    self.delfile(self.Dir+"/"+cell+"/"+x)
                elif(("Dendrite" in x or "background" in x) and self.var2.get()==1):
                    self.delfile(self.Dir+"/"+cell+"/"+x)
                elif(".json" in x and self.var3.get()==1):
                    self.delfile(self.Dir+"/"+cell+"/"+x)
                elif("ROIs.png" in x and self.var1.get()==1):
                    self.delfile(self.Dir+"/"+cell+"/"+x)
        #except:
        #    alert_popup("Warning","Oops - something went wrong")

    def delfile(self,path):
        """Function that does the deleting"""

        if os.path.exists(path):
            os.remove(path)

        else:
            alert_popup("Warning","The file: " + path + " doesn't exist",w=500,h=100)

    def close_windows(self):
        self.master.destroy()

"""========================================================================================"""

def alert_popup(title, message,w=None,h=None):
    """Generate a pop-up window for special messages."""
    root = Toplevel()
    root.title(title)
    m = message
    m += '\n'
    w = Label(root, text=m)
    w.grid(row=0,column=0,padx=5, pady=5)
    b1 = Button(root, text="OK", command=root.destroy, width=10)
    b1.grid(row=1,column=0,padx=5, pady=5)


if __name__ == '__main__':

    Mw = MainWindow()
