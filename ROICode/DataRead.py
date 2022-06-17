import numpy as np

from skimage.registration import phase_cross_correlation
import tifffile as tf
from skimage.feature import canny

import math
import os
import re

from DendriteSelector import *
from SynapseSelector import *
from zStackSelector import *
from Utility import *

from SynapseFuncs import *
from PunctaDetection import *

import matplotlib.pyplot as plt
import json

def Measure_BG(tiff_Arr_m,FileLen,z_type):
    
    """
    Input:
            tiff_Arr_m (np.array of doubles): Pixel values of all the tiff files
            FileLen                         : Number of files
            NaNlist                         : Entries where the correct file is not available
    Output:
            bg_list (np.array of doubles): values of background

    Function:
            Finds 4 corners of image and works out average, using this as background
            and kicks out any values which are 2 x the others
    """

    width = 20
    pt1 = [20,20]
    if(FileLen>1):
        bg_list = []
        for i in range(FileLen):
            bgMeasurement1 =  []
            bgMeasurement2 =  []
            bgMeasurement3 =  []
            bgMeasurement4 =  []
            
            for ii in range(20+width):
                for jj in range(20+width):
                    if(((ii-pt1[0])**2+(jj-pt1[1])**2)<width**2):
                        bgMeasurement1.append(tiff_Arr_m[i,ii,jj])
                        bgMeasurement2.append(tiff_Arr_m[i,ii,tiff_Arr_m.shape[-1]-jj])
                        bgMeasurement3.append(tiff_Arr_m[i,tiff_Arr_m.shape[-2]-ii,jj])
                        bgMeasurement4.append(tiff_Arr_m[i,tiff_Arr_m.shape[-2]-ii,tiff_Arr_m.shape[-1]-jj])


            bg = np.array([np.mean(bgMeasurement1),np.mean(bgMeasurement2),np.mean(bgMeasurement3),np.mean(bgMeasurement4)])
            bg = np.array(bg.min())
            
            #Alternative, take average of 4 corners
            #bg = bg[~(np.min(bg)*2<bg)].mean()
            
            bg_list.append(bg.min())

        return bg_list
    else:
        bgMeasurement1 =  []
        bgMeasurement2 =  []
        bgMeasurement3 =  []
        bgMeasurement4 =  []
        
        for ii in range(20+width):
            for jj in range(20+width):
                if(((ii-pt1[0])**2+(jj-pt1[1])**2)<width**2):
                    bgMeasurement1.append(tiff_Arr_m[0,ii,jj])
                    bgMeasurement2.append(tiff_Arr_m[0,ii,tiff_Arr_m.shape[-1]-jj])
                    bgMeasurement3.append(tiff_Arr_m[0,tiff_Arr_m.shape[-2]-ii,jj])
                    bgMeasurement4.append(tiff_Arr_m[0,tiff_Arr_m.shape[-2]-ii,tiff_Arr_m.shape[-1]-jj])
               
        bg = np.array([np.mean(bgMeasurement1),np.mean(bgMeasurement2),np.mean(bgMeasurement3),np.mean(bgMeasurement4)])
        bg = np.array(bg.min())

        #Alternative, take average of 4 corners
        #bg = bg[~(np.min(bg)*2<bg)].mean()

        return bg

def GetTiffData(File_Names,scale,Times=[],z_type=np.sum,Dir=None,Channels=False):

    """
    Input:
            File_Names (array of Strings): Holding name of timesteps
            scale (double)               : Pixel to μm?
            Dir (String)                 : Super directory we are looking at
            zStack (Bool)                : Flag wether we are looking at zstacks
            as_gray (Bool)               : Flag wether we want grayscale or not

    Output:
            tiff_Arr (np.array of doubles): Pixel values of all the tiff files

    Function:
            Uses tiff library to get values
    """

    Times = []

    if(File_Names==None):
       File_Names,Times = CheckFiles(Dir)

    md = getMetadata(Dir+"/"+File_Names[0])

    if File_Names[0].endswith(".lsm"):
        scale = getScale(Dir+"/"+File_Names[0])
    else:
        scale = scale

    tiff_Arr = []
    for i,x in enumerate(File_Names):
        md = getMetadata(Dir+"/"+x)
        temp = tf.imread(Dir+x)
        temp_mod = temp.reshape(md[1:])
        if(not Channels):
            temp_mod = z_type(temp_mod,axis=1,keepdims=True)
        tiff_Arr.append(z_type(temp_mod,axis=0))

    md[0] = len(tiff_Arr)
    if(not z_type==None):
        md[1] = 1
    md[2:] = tiff_Arr[0].shape
      
    return np.array(tiff_Arr),Times,md,scale

def getMetadata(filename,frame=None):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from file
    """

    if (filename.endswith('.tif')):
        return getTifDimenstions(filename)
    elif(filename.endswith('.lsm')):
       return  getLSMDimensions(filename)
    else:
        if(frame is None):
            print("Unsupported file format found. contact admin")
        #TODO: Format print as pop-up/In the main window
        exit();

def getScale(filename):
    tf_file = tf.TiffFile(filename)
    if (filename.endswith('.tif')):
        return 0.114
    elif(filename.endswith('.lsm')):
       return  tf_file.lsm_metadata['ScanInformation']['SampleSpacing']
    else:
        print("Unsupported file format found. contact admin")
        #TODO: Format print as pop-up/In the main window
        exit();

def getTifDimenstions(filename):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from tif file
    """
    try:
        meta_data = np.ones((5)) # to hold # of (t,z,c,y,x)
        tf_file = tf.TiffFile(filename)
    
        if 'slices' in tf_file.imagej_metadata.keys():
            meta_data[1] = tf_file.imagej_metadata['slices']
        if 'channels' in tf_file.imagej_metadata.keys():
            meta_data[2] = tf_file.imagej_metadata['channels']
        if 'time' in tf_file.imagej_metadata.keys():
            meta_data[0] = tf_file.imagej_metadata['time']
    
        d = tf_file.asarray()
        meta_data[3] = d.shape[-2]
        meta_data[4] = d.shape[-1]
    except:
        temp = tf.imread(filename)
        meta_data[1] = temp.shape[0]
        meta_data[2] = 1
        meta_data[0] = 1
        meta_data[3] = temp.shape[-2]
        meta_data[4] = temp.shape[-1]

    return meta_data.astype(int)

def getLSMDimensions(filename):

    """
    Input:
            filename (string) : Name of file to be read

    Output:
            meta_data (int)   : File MetaData

    Function:
            Get meta_data such as dims from lsm file
    """

    meta_data = np.ones((5))
    lsm_file = tf.TiffFile(filename)

    meta_data[0] = lsm_file.lsm_metadata['DimensionTime']
    meta_data[1] = lsm_file.lsm_metadata['DimensionZ']
    meta_data[2] = lsm_file.lsm_metadata['DimensionChannels']
    meta_data[3] = lsm_file.lsm_metadata['DimensionY']
    meta_data[4] = lsm_file.lsm_metadata['DimensionX']


    return meta_data.astype(int)

def CheckFiles(Dir):

    """
    Input:
            Dir (String)                 : Super directory we are looking at

    Output:
            Full_Time (list of strings)  : Available files in directory

    Function:
            Checks if files ending with tif or lsm are in the folder and then augments 
            the list of files with necessary ones
    """

    File_Names = []
    for x in os.listdir(Dir):
        if('.lsm' in x or '.tif' in x):
            File_Names.append(x)

    regex = re.compile('.\d+')
    File_Names_int = [ re.findall(regex,f)[0] for f in File_Names] 

    try:
        try:
            File_Names_int = [ int(f) for f in File_Names_int] 
        except:
            File_Names_int = [ int(f[1:]) for f in File_Names_int] 
        File_Names = [x for _, x in sorted(zip(File_Names_int, File_Names))]

    except:
        pass
    File_Names_int.sort()

    return File_Names,File_Names_int

def GetTiffShift(tiff_Arr,SimVars):

    """
    Input:
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters

    Output:
            tiff_arr (np.array) : The shift tiff_arr so that all snapshots overlap

    Function:
            Does an exhaustive search to find the best fitting shift and then applies
            the shift to the tiff_arr
    """

    Dir       = SimVars.Dir

    nSnaps = SimVars.Snapshots
    if(os.path.isfile(Dir+"MinDir.npy")==True):
        MinDirCum=np.load(Dir+"MinDir.npy")
    else:
        MinDir = np.zeros([2,nSnaps-1])
        if not(SimVars.frame==None):
            SimVars.frame.v.set(SimVars.frame.v.get()+"We are computing the overlap vector, it may take a bit! \n")
            SimVars.frame.master.update()
        for  t in range(nSnaps-1):
            shift, _, _ =  phase_cross_correlation(tiff_Arr[t,0,:,:], tiff_Arr[t+1,0,:,:])
            MinDir[:,t] = -shift


        MinDirCum = np.cumsum(MinDir,1)
        MinDirCum = np.insert(MinDirCum,0,0,1)
        np.save(Dir+"MinDir.npy", MinDirCum)
        
    MinDirCum = MinDirCum.astype(int)
    
    return ShiftArr(tiff_Arr,MinDirCum)

def ShiftArr(tiff_Arr,MinDirCum):

    """
    Input:
            tiff_Arr  (np.array)    : The pixel values of the of tiff files
            MinDirCum (np.array)    : The shifting directions
        
    Output:
            tiff_arr (np.array) : The shift tiff_arr so that all snapshots overlap

    Function:
            Application of MinDirCum to tiff_Arr
    """


    xLim = [(np.min(MinDirCum,1)[0]-1),(np.max(MinDirCum,1)[0]+1)]
    yLim = [(np.min(MinDirCum,1)[1]-1),(np.max(MinDirCum,1)[1]+1)]


    tiff_Arr_m = np.array([tiff_Arr[i,:,-xLim[0]+MinDirCum[0,i]:-xLim[1]+MinDirCum[0,i]
                                    ,-yLim[0]+MinDirCum[1,i]:-yLim[1]+MinDirCum[1,i]] for i in range(tiff_Arr.shape[0])])

    return tiff_Arr_m

def Measure(SynArr,tiff_Arr,SimVars):
    
    """
    Input:
            SynArr  (list of synapses)
            tiff_Arr  (np.array)    : The pixel values of the of tiff files
            MinDirCum (np.array)    : The shifting directions
        
    Output:
            None

    Function:
            Function to decide if we should apply the circular measure or the 
            shape measure
    """
    
    if not(SimVars.frame==None):
        SimVars.frame.v.set(SimVars.frame.v.get()+"Now measuring the Synapses \n")
        SimVars.frame.master.update()

    for i,S in enumerate(SynArr):

        if(SimVars.Channels>1):
            for i in range(SimVars.Channels):
                MeasureShape(S,tiff_Arr[:,i,:,:],SimVars)
        else:
            MeasureShape(S,tiff_Arr[:,0,:,:],SimVars)
            if((not SimVars.frame==None )):
                SimVars.frame.progress['value'] = (i/len(SynArr))*100
                SimVars.frame.master.update()

    return 0

def MeasureShape(S,tiff_Arr,SimVars):

    """
    Input:
            S (Synapse)
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SimVars  (class)    : The class holding all simulation parameters
    Output:
            None

    Function:
            Finds the relevant places in the tiff file and measures these for each synapse
    """
    SynA = np.array(S.xpert)
    for i in range(SimVars.Snapshots):
        Measurement = [] 
        if(SynA.ndim==2):
            SynL = np.array(S.xpert)+S.shift[i][::-1]
        elif(SynA.ndim==3):
            SynL = np.array(S.xpert)[i,:,:]+S.shift[i][::-1]
        
        if((not SimVars.frame==None)):
            SimVars.frame.progress['value'] = (i/SimVars.Snapshots)*100
            SimVars.frame.master.update()

        if(SynL.ndim==2):

            min_ii = np.floor(np.min(SynL[:,0])).astype(int)
            max_ii = np.ceil(np.max(SynL[:,0])).astype(int)   
            min_jj = np.floor(np.min(SynL[:,1])).astype(int)
            max_jj = np.ceil(np.max(SynL[:,1])).astype(int)   
            for ii in range(min_ii,max_ii):
                for jj in range(min_jj,max_jj):
                    sAng = 0
                    for x,y in zip(SynL,np.roll(SynL,2)):
                        try:
                            sAng += getAngle(x,np.array([ii,jj]),y)
                        except:
                            sAng += 0
                    if (abs(sAng-2*np.pi)<1e-3):
                        try:
                            Measurement.append(tiff_Arr[i,jj,ii])
                        except:
                            Measurement.append(tiff_Arr[jj,ii])
            try:
                S.area.append(len(Measurement)*SimVars.Unit**2)
                S.max.append(int(np.max(Measurement)))
                S.min.append(int(np.min(Measurement)))
                S.RawIntDen.append(int(np.sum(Measurement)))
                S.IntDen.append(np.float64(np.sum(Measurement))*len(Measurement)*SimVars.Unit**2)
                S.mean.append(np.float64(np.mean(Measurement)))
            except:
                S.area.append(math.nan)
                S.mean.append(math.nan)
                S.max.append(math.nan)
                S.min.append(math.nan)
                S.RawIntDen.append(math.nan)
                S.IntDen.append(math.nan)


    return 0
    
def PlotSyn(tiff_Arr,SynArr,SimVars,Mode=''):

    """
    Input:
            
            tiff_Arr (np.array) : The pixel values of the of tiff files
            SynArr (np.array of Synapses) : Array holding synaps information
            SimVars  (class)    : The class holding all simulation parameters

    Output:
            N/A
    Function:
            Plots Stuff
    """ 

    plt.subplots(figsize=(12,12))
    try:
        plt.imshow(tiff_Arr)
    except:
       plt.imshow(tiff_Arr.max(axis=0))

    colors = {0:'k',1:'r'}
    for i,Synapse in enumerate(SynArr):
        try:
            xplt = np.vstack([Synapse.xpert,Synapse.xpert[0]])
        except:
            xplt = np.vstack([Synapse.xpert[0],Synapse.xpert[0][0]])

        plt.plot(xplt[:,0],xplt[:,1],colors[Synapse.FlagUp])
        plt.annotate(str(i),(Synapse.location[0]+5,Synapse.location[1]+5),color=colors[Synapse.FlagUp])

    plt.tight_layout()
    plt.savefig(SimVars.Dir+"ROIs"+Mode+".png")
    plt.close()

def FullEval(Dir,Mode,Channels,Unit=1,z_type='SUM',frame=None):
    
    #TODO: Describe function
    
    SimVars = Simulation(Unit,0,Dir,1,Mode,z_type,frame)
    
    # Get images
    tiff_Arr,SimVars.Times,meta_data,scale = GetTiffData(None,Unit,[],SimVars.z_type,SimVars.Dir,Channels=Channels)
    # Set parameters
    SimVars.Unit = scale
    SimVars.Snapshots = meta_data[0]
    SimVars.Channels  = meta_data[2]
    SimVars.bgmean = np.zeros([SimVars.Snapshots,SimVars.Channels])

    # Get shifting of snapshots
    if(SimVars.Snapshots>1):
        tiff_Arr          = GetTiffShift(tiff_Arr, SimVars)
        

    DM = DendriteMeasure(tiff_Arr[0,:,:,:],SimVars)

    # Get Background values
    cArr_m = np.zeros_like(tiff_Arr[0,:,:,:])
    for i in range(SimVars.Channels):
        cArr_m[i,:,:] = canny(tiff_Arr[:,i,:,:].max(axis=0),sigma=1)
        SimVars.bgmean[:,i] = Measure_BG(tiff_Arr[:,i,:,:],SimVars.Snapshots,SimVars.z_type)
    
    # Do dendrite calculation
    DendArr,length,Somas,selected_channel = DM.Automatic(SimVars)

    if(SimVars.Mode=="Puncta"):
        punctas = FullEval_Puncta(SimVars,tiff_Arr,Somas,DendArr)
        
    else:
        
        sigma = 3
        dend_stats = {}
        max_dend_stats = {}
        dend_info = {}
        
        # Get dendrite statistics
        if(DM.ShowStats):
            for dxd,dend in enumerate(DendArr):
                dend_stats["Dendrite"+str(dxd)],max_dend_stats["Dendrite"+str(dxd)], dend_info = DM.GetAdaptiveWidths(dend,sigma)
                np.save(Dir+"dend_info_"+str(dxd)+".npy",dend_info)
            
        with open(Dir+'dend_stat.json', 'w') as fp: json.dump(dend_stats,fp,indent=4)
        fp.close()
        with open(Dir+'max_dend_stat.json', 'w') as fp: json.dump(max_dend_stats,fp,indent=4)
        fp.close()

        # Run the Synapse builder
        SynArr = []

        SyBuild = SynapseBuilder(SynArr,True,cArr_m[0,:,:],tiff_Arr[0,:,:,:],DendArr,SimVars,selected_channel)
        SyBuild.ptI.pts = SyBuild.ptI.pts[SyBuild.ptI.score>SyBuild.ptI.sigma]
        SyBuild.ptI.flag = SyBuild.ptI.flag[SyBuild.ptI.score>SyBuild.ptI.sigma]
        tROIs = []
        for R,S in zip(SyBuild.ptI.ROIs, SyBuild.ptI.score):
            if(S>SyBuild.ptI.sigma):
                tROIs.append(R)

        SyBuild.ptI.ROIs = tROIs
        if(SimVars.Mode=="Area"):
            Synapses = FullEval_Area(SimVars,tiff_Arr,cArr_m,DendArr,SyBuild)
            
        elif(SimVars.Mode=="Luminosity" or SimVars.Mode=="Soma"):
            Synapses = FullEval_Lum(SimVars,tiff_Arr,cArr_m,DendArr,SyBuild)
        
        Synapses = SynDistance(Synapses,DendArr,SimVars.Unit,SimVars.Mode)
        
        RC = RoiCorrector(Synapses,DendArr,tiff_Arr,SimVars)
        Synapses = RC.SynArr
        selected_channel = RC.Channel
        Measure(Synapses,tiff_Arr,SimVars)

        SaveSynDict(Synapses,SimVars.bgmean,SimVars.Dir,SimVars.Mode)

        if not(SimVars.frame==None):
            SimVars.frame.v.set(SimVars.frame.v.get()+"Done with "+ os.path.split(Dir[:-1])[-1]+" \n")
            SimVars.frame.master.update()

        PlotSyn(tiff_Arr[0,selected_channel,:,:],Synapses,SimVars)

    return 0

def FullEval_Puncta(SimVars,tiff_arr,somas,DendArr):

    #TODO: Describe function
    
    Dend_dict = {}
    Soma_dict = {}
    for sdx,s in enumerate(somas[:]):
        Soma_dict["soma_"+str(sdx)] = s
    for d in range(0,len(DendArr)):
        Dend_dict["dendrite_"+str(d)] = DendArr[d]

    
    half_width = [2.5,5,7.5]
    channels = [1,3]

    for w in half_width:
        PD = PunctaDetection(SimVars,tiff_arr,Soma_dict,Dend_dict,channels,w)
        somatic_punctas , dendritic_punctas = PD.GetPunctas()
        for  c in somatic_punctas.keys():

            op_dir = SimVars.Dir+str(c)+"/"+str(2*w)+"/"
            os.makedirs(op_dir, exist_ok=True)
            with open(op_dir+"soma_puncta_channel"+str(c)+"_"+str(2*w)+".json",'w') as f:
                json.dump(somatic_punctas[c], f,sort_keys=True, indent=4)
            with open(op_dir+"dend_puncta_channel"+str(c)+"_"+str(2*w)+".json",'w') as f:
                json.dump(dendritic_punctas[c], f,sort_keys=True, indent=4)
        
    print("puncta detection complete")
    return 0

def FullEval_Lum(SimVars,tiff_Arr,cArr_m,DendArr,SyBuild):

    #TODO: Describe function
    
    Synapses = []
    if not(SimVars.frame==None):
        SimVars.frame.v.set(SimVars.frame.v.get()+"Working out the synapse ROIs! \n")
        SimVars.frame.master.update()

    for i,(S,T,R) in enumerate(zip(SyBuild.ptI.pts,SyBuild.ptI.flag,SyBuild.ptI.ROIs)):
        if(T==1):
            t='Stim'
        else:
            t='Spine'
        Syn = Synapse(loc=np.round(S).astype(int).tolist(),radloc=[0,0],nSnaps=0,stack=0,Unit=SimVars.Unit,Syntype=t)
        Syn.Times  = SimVars.Times
        if(len(R)==0):
            Syn.xpert,_,Syn.shift = FindShape(tiff_Arr[:,SyBuild.channel_selected,:,:],np.array(Syn.location),DendArr,np.round(SyBuild.ptI.pts).astype(int),
                SimVars.bgmean[:,0].max(),True)
        else:
            pt = np.round(S).astype(int)
            Syn.xpert = R
            tiff_Arr_small = tiff_Arr[:,SyBuild.channel_selected,max(pt[1]-50,0):min(pt[1]+50,tiff_Arr.shape[-2]),max(pt[0]-50,0):min(pt[0]+50,tiff_Arr.shape[-1])]
            Syn.shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()   

        if(not SimVars.frame==None ):
            SimVars.frame.progress['value'] = (i/len(SyBuild.ptI.pts))*100
            SimVars.frame.master.update()

        Synapses.append(Syn)

    return Synapses
    

def FullEval_Area(SimVars,tiff_Arr,cArr_m,DendArr,SyBuild):
    
    #TODO: Describe function
    
    Synapses = []
   
    if not(SimVars.frame==None):
        SimVars.frame.v.set(SimVars.frame.v.get()+"Working out the synapse ROIs! \n")
        SimVars.frame.master.update()
        
    for j,(S,T) in enumerate(zip(SyBuild.ptI.pts,SyBuild.ptI.flag)):
        if(T==1):
            t='Stim'
        else:
            t='Spine'
        Syn = Synapse(loc=np.round(S).astype(int).tolist(),radloc=[0,0],nSnaps=0,stack=0,Unit=SimVars.Unit,Syntype=t,xpert = [])
        Syn.Times  = SimVars.Times
        if(tiff_Arr.ndim>2):
            Syn.area = []
            pt = Syn.location
            tiff_Arr_small = tiff_Arr[:,0,max(pt[1]-50,0):min(pt[1]+50,tiff_Arr.shape[-2]),max(pt[0]-50,0):min(pt[0]+50,tiff_Arr.shape[-1])]
            Syn.shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()   
            for i in range(SimVars.Snapshots):
                xpert,_,_ = FindShape(tiff_Arr[i,SyBuild.channel_selected,:,:],np.array(Syn.location),DendArr,np.round(SyBuild.ptI.pts).astype(int),SimVars.bgmean[i,0],True)
                Syn.xpert.append(xpert)
        else: 
            Syn.xpert,_,_ = FindShape(tiff_Arr,np.array(Syn.location),DendArr,np.round(SyBuild.ptI.pts).astype(int),SimVars.bgmean[0,0],True)
            Syn.xpert.append(xpert)
            
        if(not SimVars.frame==None ):
            SimVars.frame.progress['value'] = (j/len(SyBuild.ptI.pts))*100
            SimVars.frame.master.update()

        Synapses.append(Syn)

    return Synapses

def EvalTot(Dir,Mode):

    #TODO: Describe function
    
    frame=None
    Unit=0.066
    z_type='Max'

    if(Mode=="Luminosity"):

        for d in os.listdir(Dir):
            try:
                if('Synapse_l.json' in os.listdir(Dir+d)):
                    print('='*30)
                    print('Doing ' + d)
                    
                    SimVars = Simulation(Unit,0,Dir+d+'/',1,Mode,z_type,frame)

                    DendArr,_,_ = LoadDend(SimVars.Dir)
                    tiff_Arr,SimVars.Times,meta_data,scale = GetTiffData(None,Unit,[],SimVars.z_type,SimVars.Dir,Channels=False)
                    SimVars.CheckLims = min(np.min(np.shape(tiff_Arr[:,:,0,0]))//4,100)
                    SimVars.Unit = scale
                    SimVars.Snapshots = meta_data[0]
                    SimVars.Channels  = meta_data[2]
                    SimVars.bgmean = np.zeros([SimVars.Snapshots,SimVars.Channels])
                    for i in range(SimVars.Channels):
                        SimVars.bgmean[:,i] = Measure_BG(tiff_Arr[:,i,:,:],SimVars.Snapshots,SimVars.z_type)

                    if(SimVars.Snapshots>1):
                        tiff_Arr          = GetTiffShift(tiff_Arr, SimVars)
                    SynArr = ReadSynDict(SimVars.Dir,0,SimVars.Unit,'Luminosity')
                    other_pts = np.round([S.location for S in SynArr]).astype(int)
                    Measure(SynArr,tiff_Arr,SimVars)
                    SaveSynDict(SynArr,SimVars.bgmean,SimVars.Dir,SimVars.Mode)
                    print('Success')
            except:
                pass
    elif(Mode=="Area"):
        for d in os.listdir(Dir):
            try:
                if('Synapse_l.json' in os.listdir(Dir+d)):
                    print('='*30)
                    print('Doing ' + d)
                    
                    SimVars = Simulation(Unit,0,Dir+d+'/',1,Mode,z_type,frame)

                    DendArr,_,_ = LoadDend(SimVars.Dir)
                    tiff_Arr,SimVars.Times,meta_data,scale = GetTiffData(None,Unit,[],SimVars.z_type,SimVars.Dir,Channels=False)
                    SimVars.CheckLims = min(np.min(np.shape(tiff_Arr[:,:,0,0]))//4,100)
                    SimVars.Unit = scale
                    SimVars.Snapshots = meta_data[0]
                    SimVars.Channels  = meta_data[2]
                    SimVars.bgmean = np.zeros([SimVars.Snapshots,SimVars.Channels])
                    for i in range(SimVars.Channels):
                        SimVars.bgmean[:,i] = Measure_BG(tiff_Arr[:,i,:,:],SimVars.Snapshots,SimVars.z_type)
                    
                    if(SimVars.Snapshots>1):
                        tiff_Arr          = GetTiffShift(tiff_Arr, SimVars)
                    SynArr = ReadSynDict(SimVars.Dir,0,SimVars.Unit,Mode)
                    other_pts = np.round([S.location for S in SynArr]).astype(int)
                    
                    for j,Syn in enumerate(SynArr):
                        pt = Syn.location
                        tiff_Arr_small = tiff_Arr[:,0,max(pt[1]-50,0):min(pt[1]+50,tiff_Arr.shape[-2]),max(pt[0]-50,0):min(pt[0]+50,tiff_Arr.shape[-1])]
                        Syn.shift = SpineShift(tiff_Arr_small).T.astype(int).tolist()   
                        Syn.xpert = []
                        Syn.area = []
                        Syn.Times  = SimVars.Times
                        for i in range(SimVars.Snapshots):
                            xpert,_,_ = FindShape(tiff_Arr[i,0,:,:],np.array(Syn.location),DendArr,other_pts,SimVars.bgmean[i,0],True)
                            Syn.xpert.append(xpert)
                        Syn.min = []
                        Syn.max = []
                        Syn.mean = []
                        Syn.RawIntDen = []
                        Syn.IntDen = []   
                        
                    Measure(SynArr,tiff_Arr,SimVars)
                    SaveSynDict(SynArr,SimVars.bgmean,SimVars.Dir,SimVars.Mode)
                    print('Success')
            except:
                pass

if __name__ == '__main__':
    
    Mode = "Area"
    print('@'*30)
    print('Doing 1')
    for d in ['Control/','Anisomycin/','CHX/','CamKII/','Calcineurin/','Sham/','CamKIIb/']:
        print('Now on: '+d)
        Mode = "Luminosity"
        print('-'*30)
        print('Luminosity')
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_1spine/"+d,Mode)
        print('-'*30)
        print('Area')
        Mode = "Area"
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_1spine/"+d,Mode)
    print('@'*30)
    print('Doing 3')
    for d in ['Control/','Sham/']:
        print('Now on: '+d)
        Mode = "Luminosity"
        print('-'*30)
        print('Luminosity')
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_3spine/"+d,Mode)
        print('-'*30)
        print('Area')
        Mode = "Area"
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_3spine/"+d,Mode)
    print('@'*30)
    print('Doing 7')
    for d in ['Control/','Anisomycin/','CHX/','CamKII/','Calcineurin/','Sham/','Distributed/','CamKIIb/']:
        print('Now on: '+d)
        Mode = "Luminosity"
        print('-'*30)
        print('Luminosity')
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_7spine/"+d,Mode)
        print('-'*30)
        print('Area')
        Mode = "Area"
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_7spine/"+d,Mode)
    print('@@@@@@@@@@@@@@@@')
    print('Doing 15')
    for d in ['Control/','Sham/']:
        print('Now on: '+d)
        Mode = "Luminosity"
        print('-'*30)
        print('Luminosity')
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_15spine/"+d,Mode)
        print('-'*30)
        print('Area')
        Mode = "Area"
        EvalTot("/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/MaxProj/CA1_15spine/"+d,Mode)

    # Dir,Mode,Channels = '/Users/maximilianeggl/Downloads/Test/cell_0/','Luminosity',False
    # Dir,Mode,Channels = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/Sham/cell_1/','Luminosity',False
    # Dir,Mode,Channels  = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/Code/HeteroPlast/TestImages/cell_2/','Soma',True
    # FullEval(Dir,'Luminosity',Channels,Unit=0.114,z_type='Max',frame=None)

    
    