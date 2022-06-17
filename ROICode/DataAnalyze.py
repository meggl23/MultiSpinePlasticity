import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons,TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import scipy.stats as stats
import matplotlib.patches as mpatches
import json

import math

import os.path

def GetData(Dir,AcceptableDict):
    """Function that obtains the necessary data from csv files and returns it in the right shape"""


    with open(Dir+'Synapse.json', 'r') as fp: temp = json.load(fp)
    bg= np.load(Dir+'background.npy')

    dist = []
    Dat  = []
    
    for t in temp:
        Dat.append(t["mean"])
        dist.append(t["distance"])
    return Dat,dist,bg,len(bg)


   # stim = np.empty([len(S),lenTime])

    # for i in range(len(S)):
    #     stim[i,:] = S[i][0][MeanVis:MeanVis+lenTime]

    # A = np.vstack([np.asarray(c)[1:,:] for c in S])
    # AComp = (A[:,MeanVis:MeanVis+lenTime]-A[:,-2*lenTime:-lenTime])
    # dist = np.hstack([np.asarray(c)[1:,3] for c in S])

#    return [stim,A,AComp,dist],lenTime


"""========================================================================================"""


class ContourWindow:
    """Class that defines the contour plot analysis window"""
    def __init__(self,Data,dist,bg,Dir,AcceptableDict,lenTime,contNum=None):
        
        self.Dir = Dir
        cells = np.array([g for g in AcceptableDict.keys()])
        truths = [AcceptableDict[c] for c in AcceptableDict.keys()]
        
        MeanVis = 4
        #MeanVis = 4
        self.cN = contNum
        if(all(truths)):
            self.cellname = '_all'

        else:

            self.cellname=""
            for c in cells[truths]:
                self.cellname+="_"+c
        self.time = [-15,-10,-5,2]
        self.time.extend([i*10 for i in range(1,lenTime-3)])

        self.Unnorm   = np.copy(Data)
        self.Norm = self.Unnorm / np.nanmean(self.Unnorm[:,:3],1)[:,np.newaxis]
        self.UnnormBg = self.Unnorm - bg
        self.NormBg = self.UnnormBg/np.nanmean(self.UnnormBg[:,:3],1)[:,np.newaxis]
        self.dist  = np.copy(dist)


        if not(contNum==None):
            dbuck = np.max(self.dist)/contNum
            uTemp = np.zeros([contNum,self.Unnorm.shape[-1]])
            nTemp = np.zeros([contNum,self.Norm.shape[-1]])
            nbgTemp = np.zeros([contNum,self.NormBg.shape[-1]])
            for i in range(contNum):
                uTemp[i,:] = np.nanmean(self.Unnorm[np.logical_and(self.dist>i*dbuck,self.dist<(i+1)*dbuck),:],0)
                nTemp[i,:] = np.nanmean(self.Norm[np.logical_and(self.dist>i*dbuck,self.dist<(i+1)*dbuck),:],0)
                nbgTemp[i,:] = np.nanmean(self.NormBg[np.logical_and(self.dist>i*dbuck,self.dist<(i+1)*dbuck),:],0)
            self.Unnorm = np.copy(uTemp)
            self.Norm = np.copy(nTemp)
            self.NormBg = np.copy(nbgTemp)
            self.T,self.D = np.meshgrid(self.time,np.linspace(0,np.max(self.dist),contNum))
        else:
             self.T,self.D = np.meshgrid(self.time,self.dist)
        axcolor = 'lightgoldenrodyellow'
        
        
        self.fig, self.ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.45, bottom=0.20)

        self.ax.margins(x=0)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Distance')

        self.SpineData = self.Unnorm.T

        im = self.ax.contourf(self.D,self.T,self.SpineData.T,500,cmap='inferno')

        self.cax = make_axes_locatable(self.ax).append_axes("right", size="5%", pad="2%")
        self.cb = self.fig.colorbar(im, cax=self.cax)

        rax = plt.axes([0.025, 0.4, 0.35, 0.23], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('Unnormalized', 'Normalized', 'Norm-background'), active=0,activecolor='black')
        self.radio.on_clicked(self.modefunc)
        
        saveax = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.sbutton = Button(saveax, 'Save', hovercolor='0.5')
        self.sbutton.on_clicked(self.save)
        
        self.ax.set_ylabel('Time (min)')
        self.ax.set_xlabel('Distance from stim spine (μm)')
        self.ax.set_xlim([0,20])
        self.ax.set_ylim([0,40])

        self.tchoose = 0
        self.tax1 = plt.axes([0.025, 0.9, 0.1, 0.04], facecolor='0.6')
        self.tbutton1 = Button(self.tax1, 'Contour')
        self.tbutton1.on_clicked(self.switchtab)
        self.tax2 = plt.axes([0.145, 0.9, 0.1, 0.04], facecolor='0.9')
        self.tbutton2 = Button(self.tax2, 'Histogram')
        self.tbutton2.on_clicked(self.switchtab)
        plt.show()


    def modefunc(self,label):
        """Function that changes the data and replots the contourplot"""
        if(label=="Unnormalized"):
           self.SpineData = self.Unnorm.T

        elif(label=="Normalized"):
           self.SpineData = self.Norm.T

        elif(label=='Norm-background'):
           self.SpineData = self.NormBg.T

        self.ax.clear()
        self.cb.remove()
        if(self.tchoose==0):
            im = self.ax.contourf(self.D,self.T,self.SpineData.T,500,cmap='inferno')
        else:
        #    if(self.cN==None):
        #        print(self.dist)
        #        im = self.ax.pcolor(self.dist,self.time,self.SpineData,shading='nearest')
        #    else:
        #        im = self.ax.pcolor(np.linspace(0,np.max(self.dist),self.cN),self.time,self.SpineData,shading='nearest')    
            im = self.ax.imshow(np.flip(self.SpineData,0), cmap=cm.jet, interpolation='nearest',aspect="auto",extent=[min(self.dist),max(self.dist),-15,40])
        self.cax = make_axes_locatable(self.ax).append_axes("right", size="5%", pad="2%")
        self.cb = self.fig.colorbar(im, cax=self.cax)

        self.ax.set_ylabel('Time (min)')
        self.ax.set_xlabel('Distance from stim spine (μm)')
        self.ax.set_xlim([0,20])
        self.ax.set_ylim([0,40])
        plt.draw()

    def save(self,event):
        """Function that saves the contour plot with the right name"""
        self.sbutton.ax.set_facecolor('white')
        self.sbutton.ax.figure.canvas.draw()

        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        name = self.Dir +'/'+ self.radio.value_selected+self.cellname +'_contour.png'

        plt.savefig(name,format='png',bbox_inches=extent.expanded(1.3, 1.3))
        plt.pause(2.0)
        self.sbutton.ax.set_facecolor(self.sbutton.color)
        self.sbutton.ax.figure.canvas.draw()

    def switchtab(self,event):
        """Function that saves the contour plot with the right name"""
        if event.inaxes == self.tax1: 
            if(self.tchoose==1):
                self.tchoose=0
                self.ax.clear()
                self.cb.remove()
                
                im = self.ax.contourf(self.D,self.T,self.SpineData.T,500,cmap='inferno')

                self.cax = make_axes_locatable(self.ax).append_axes("right", size="5%", pad="2%")
                self.cb = self.fig.colorbar(im, cax=self.cax)

                self.ax.set_ylabel('Time (min)')
                self.ax.set_xlabel('Distance from stim spine (μm)')
                self.ax.set_xlim([0,20])
                self.ax.set_ylim([0,40])                
                plt.draw()
        elif event.inaxes == self.tax2: 
            if(self.tchoose==0):
                self.tchoose=1
                self.ax.clear()
                self.cb.remove()
                self.fig.set_size_inches(10,5)
                #if(self.cN==None):
                #    print(self.dist)
                #    i = self.ax.pcolor(self.dist,self.time,self.SpineData,shading='nearest')
                #else:
                #    i = self.ax.pcolor(np.linspace(0,np.max(self.dist),self.cN),self.time,self.SpineData,shading='nearest')                    
                i = self.ax.imshow(np.flip(self.SpineData,0), cmap=cm.jet, interpolation='nearest',aspect="auto",extent=[min(self.dist),max(self.dist),-15,40])
                self.cax = make_axes_locatable(self.ax).append_axes("right", size="5%", pad="2%")
                self.cb = self.fig.colorbar(i, cax=self.cax)
                self.ax.set_ylabel('Time (min)')
                self.ax.set_xlabel('Distance from stim spine (μm)')
                self.ax.set_xlim([0,20])
                self.ax.set_ylim([0,40])
                plt.draw()

"""========================================================================================"""

class MultiLineWindow:

    """Class that defines the window with the bucketed lines"""
    def __init__(self,Data,dist,bg,Dir,Thresholds,AcceptableDict,lenTime):
        
        cells = np.array([g for g in AcceptableDict.keys()])
        truths = [AcceptableDict[c] for c in AcceptableDict.keys()]
        if(all(truths)):
            self.cellname = '_all'
        else:
            self.cellname=""
            for c in cells[truths]:
                self.cellname+="_"+c
        MeanVis = 4
        #MeanVis = 4
        self.time = [-15,-10,-5,2]
        self.time.extend([i*10 for i in range(1,lenTime-3)])
        self.Dir = Dir
        self.Unnorm   = np.copy(Data)
        self.Norm = self.Unnorm / np.nanmean(self.Unnorm[:,:3],1)[:,np.newaxis]
        self.UnnormBg = self.Unnorm - bg
        self.NormBg = self.UnnormBg/np.nanmean(self.UnnormBg[:,:3],1)[:,np.newaxis]
        self.dist  = np.copy(dist)
        
        axcolor = 'lightgoldenrodyellow'
        
        self.Th = Thresholds

        self.fig, self.ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.45, bottom=0.20)
        self.ax.margins(x=0)

        self.SpineData = self.Unnorm
        self.compareline = self.ax.axhline(y=1,color='black',linestyle='--',visible=False)

        self.pltlines = [self.ax.errorbar(self.time, np.nanmean(self.SpineData[self.dist<Thresholds[0]],0),
            yerr=stats.sem(self.SpineData[self.dist<Thresholds[0]],0,nan_policy='omit'),
            fmt='-o',capsize=5)]

        self.pltlines[0].set_label('stim distance<'+str(Thresholds[0]))

        tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist<Thresholds[0],:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<Thresholds[0],3:].T])
        vSig = np.array(self.time[3:]).astype(float)
        vSig[~tSig] = math.nan
        self.StatSig = [plt.plot(vSig,np.max(np.nanmean(self.SpineData))*1.45*np.ones_like(vSig),'*',color=self.pltlines[0][0].get_color())]

        for i in range(len(Thresholds)-1):

            self.pltlines.append(self.ax.errorbar(self.time, np.nanmean(self.SpineData[np.logical_and(self.dist>=Thresholds[i], self.dist<Thresholds[i+1])],0),
            yerr=stats.sem(self.SpineData[np.logical_and(self.dist>=Thresholds[i], self.dist<Thresholds[i+1])],0,nan_policy='omit'),
            fmt='-o',capsize=5,visible=True))

            self.pltlines[i+1].set_label(str(Thresholds[i])+'<stim distance<'+str(Thresholds[i+1]))

            tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[np.logical_and(self.dist>=Thresholds[i], self.dist<Thresholds[i+1]),:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[np.logical_and(self.dist>=Thresholds[i], self.dist<Thresholds[i+1]),3:].T])
            vSig = np.array(self.time[3:]).astype(float)
            vSig[~tSig] = math.nan
            self.StatSig.append(plt.plot(vSig,np.max(np.nanmean(self.SpineData))*(1.5-(i+2)*0.05)*np.ones_like(vSig),'*',color=self.pltlines[i+1][0].get_color()))

        self.pltlines.append(self.ax.errorbar(self.time, np.nanmean(self.SpineData[self.dist>Thresholds[-1]],0),
            yerr=stats.sem(self.SpineData[self.dist>Thresholds[-1]],0,nan_policy='omit'),
            fmt='-o',capsize=5,visible=True))

        self.pltlines[-1].set_label(str(Thresholds[-1])+'<stim distance')

        tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist>Thresholds[-1],:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist>Thresholds[-1],3:].T])
        vSig = np.array(self.time[3:]).astype(float)
        vSig[~tSig] = math.nan
        self.StatSig.append(plt.plot(vSig,np.max(np.nanmean(self.SpineData))*(1.5-(len(Thresholds)+1)*0.05)*np.ones_like(vSig),'*',color=self.pltlines[-1][0].get_color()))

        rax = plt.axes([0.025, 0.65, 0.35, 0.23], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('Unnormalized', 'Normalized', 'Norm-background'), active=0,activecolor='black')
        self.radio.on_clicked(self.modefunc)


        checkax = plt.axes([0.025, 0.2, 0.35, 0.4], facecolor=axcolor)
        self.label = [c.get_label() for c in self.pltlines]
        self.visibility = [True for c in self.pltlines]
        self.check = CheckButtons(checkax, self.label, self.visibility)
        self.check.on_clicked(self.func)

        saveax = plt.axes([0.6, 0.025, 0.1, 0.04])
        button = Button(saveax, 'Save', hovercolor='0.975')
        button.on_clicked(self.save)

        self.ax.legend()
        self.ax.set_ylim([np.min(np.nanmean(self.SpineData))*0.5,np.max(np.nanmean(self.SpineData))*1.5])

        saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button2 = Button(saveax,'Legend', hovercolor='0.975')
        button2.on_clicked(self.legendvis)

        self.ax.set_ylabel('Unnormalized luminosity')
        self.ax.set_xlabel('Time (min)')

        plt.show()

    def func(self,label):
        """Function that toggles visibility of chosen lines"""

        labels = np.array([c.get_label() for c in self.pltlines])

        visibilityErrbar(self.pltlines[np.where(label==labels)[0][0]])

        self.StatSig[np.where(label==labels)[0][0]][0].set_visible(not self.StatSig[np.where(label==labels)[0][0]][0].get_visible())
        
        self.ax.legend()
        
        plt.draw()

    def modefunc(self,label):
        """Function that changes the data and replots the lines"""
        if(label=="Unnormalized"):
            self.SpineData = self.Unnorm
            self.compareline.set_visible(False)
            self.ax.set_ylabel('Unnormalized luminosity')

        elif(label=="Normalized"):
            self.SpineData = self.Norm
            self.compareline.set_visible(True)
            self.ax.set_ylabel('Normalized luminosity')

        elif(label=='Norm-background'):
            self.SpineData = self.NormBg
            self.compareline.set_visible(True)
            self.ax.set_ylabel('Normalized luminosity - bg')

        update_errorbar(self.pltlines[0],self.time, np.nanmean(self.SpineData[self.dist<self.Th[0]],0),
            yerr=stats.sem(self.SpineData[self.dist<self.Th[0]],0))
        
        tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist<self.Th[0],:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<self.Th[0],3:].T])
        vSig = np.array(self.time[3:]).astype(float)
        vSig[~tSig] = math.nan

        self.StatSig[0][0].set_xdata(vSig)
        self.StatSig[0][0].set_ydata(np.max(np.nanmean(self.SpineData))*1.45*np.ones_like(vSig))

        for i in range(len(self.Th)-1):
            update_errorbar(self.pltlines[i+1],self.time, np.nanmean(self.SpineData[np.logical_and(self.dist>=self.Th[i], self.dist<self.Th[i+1])],0),
            yerr=stats.sem(self.SpineData[np.logical_and(self.dist>=self.Th[i], self.dist<self.Th[i+1])],0))
            
            tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[np.logical_and(self.dist>=self.Th[i], self.dist<self.Th[i+1]),:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[np.logical_and(self.dist>=self.Th[i], self.dist<self.Th[i+1]),3:].T])
            vSig = np.array(self.time[3:]).astype(float)
            vSig[~tSig] = math.nan

            self.StatSig[i+1][0].set_xdata(vSig)
            self.StatSig[i+1][0].set_ydata(np.max(np.nanmean(self.SpineData))*(1.5-(i+2)*0.05)*np.ones_like(vSig))

        update_errorbar(self.pltlines[-1],self.time, np.nanmean(self.SpineData[self.dist>self.Th[-1]],0),
            yerr=stats.sem(self.SpineData[self.dist>self.Th[-1]],0))
        
        tSig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist>self.Th[-1],:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist>self.Th[-1],3:].T])
        vSig = np.array(self.time[3:]).astype(float)
        vSig[~tSig] = math.nan

        self.StatSig[-1][0].set_xdata(vSig)
        self.StatSig[-1][0].set_ydata(np.max(np.nanmean(self.SpineData))*(1.5-(len(self.Th)+1)*0.05)*np.ones_like(vSig))
        
        self.ax.set_ylim([np.min(np.nanmean(self.SpineData))*0.5,np.max(np.nanmean(self.SpineData))*1.5])
        plt.draw()

    def save(self,event):
        """Function that saves the plot with the right name"""
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        name = self.Dir +'/'+ self.radio.value_selected +self.cellname+'_multiline.png'

        plt.savefig(name,format='png',bbox_inches=extent.expanded(1.3, 1.3))

    def legendvis(self,event):
        self.ax.get_legend().set_visible(not self.ax.get_legend().get_visible())
        self.fig.canvas.draw()

class DataAnalWindow:
    """Class that defines the window with the dynamic lines"""
    def __init__(self,Data,dist,bg,Dir,AcceptableDict,lenTime):

        cells = np.array([g for g in AcceptableDict.keys()])
        truths = [AcceptableDict[c] for c in AcceptableDict.keys()]
        if(all(truths)):
            self.cellname = '_all'
        else:
            self.cellname=""
            for c in cells[truths]:
                self.cellname+="_"+c
        MeanVis = 4
        #MeanVis = 4
        self.Dir = Dir
        self.time = [-15,-10,-5,2]
        self.time.extend([i*10 for i in range(1,lenTime-3)])
        
        self.Unnorm   = np.copy(Data)
        self.Norm = self.Unnorm / np.nanmean(self.Unnorm[:,:3],1)[:,np.newaxis]
        self.UnnormBg = self.Unnorm - bg
        self.NormBg = self.UnnormBg/np.nanmean(self.UnnormBg[:,:3],1)[:,np.newaxis]
        self.dist  = np.copy(dist)
        
        self.stim = self.Unnorm[self.dist==0]
        self.stimNorm = self.Norm[self.dist==0]
        self.StimNormBg =self.NormBg[self.dist==0]

        self.delta_dist = 0.1
        self.cutoff     = 4

        axcolor = 'lightgoldenrodyellow'
        
        self.fig, self.ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.45, bottom=0.20)
        self.compareline = self.ax.axhline(y=1,color='black',linestyle='--',visible=False)
        self.ax.margins(x=0)
        
        self.stimData = self.stim
        self.SpineData = self.Unnorm
        
        self.StimMean = self.ax.errorbar(self.time, np.nanmean(self.stimData,0),
            yerr=stats.sem(self.stimData,0,nan_policy='omit'),
            fmt='-o',capsize=5,visible=False,color='blue')

        _, caps, bars = self.StimMean
        for cap in caps:
            cap.set_visible(False)

        for bar in bars:
            bar.set_visible(False)

        self.stimLines = []
        for i in range(len(self.stim)):
            self.stimLines.append(self.ax.plot(self.time,self.stimData[i,:],alpha=0.1,visible=False))
        
        self.allLines = []
        for i in range(len(self.SpineData)):
            self.allLines.append(self.ax.plot(self.time,self.SpineData[i,:],alpha=0.1,color='k',visible=False))

        self.ProxMean = self.ax.errorbar(self.time, np.nanmean(self.SpineData[self.dist<self.cutoff],0),
            yerr=stats.sem(self.SpineData[self.dist<self.cutoff],0,nan_policy='omit'),
            fmt='-o',capsize=5,visible=False,color='red')
        
        _, caps, bars = self.ProxMean
        for cap in caps:
            cap.set_visible(False)
        for bar in bars:
            bar.set_visible(False)

        
        self.DistMean = self.ax.errorbar(self.time, np.nanmean(self.SpineData[self.dist>self.cutoff],0),
            yerr=stats.sem(self.SpineData[self.dist>self.cutoff],0,nan_policy='omit'),
            fmt='-o',capsize=5,visible=False,color='green')

        _, caps, bars = self.DistMean
        for cap in caps:
            cap.set_visible(False)
        for bar in bars:
            bar.set_visible(False)

        plt.draw()

        self.stimMeansig = np.array([0.05>stats.ttest_ind(t,self.stimData[:,:3].flatten(),nan_policy='omit').pvalue for t in self.stimData[:,3:].T])
        self.proxMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist<self.cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<self.cutoff,3:].T])
        self.distMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist>self.cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist>self.cutoff,3:].T])

        stimMeanVec = np.array(self.time[3:]).astype(float)
        proxMeanVec = np.array(self.time[3:]).astype(float)
        distMeanVec = np.array(self.time[3:]).astype(float)

        stimMeanVec[~self.stimMeansig] = math.nan
        proxMeanVec[~self.proxMeansig] = math.nan
        distMeanVec[~self.distMeansig] = math.nan

        self.lSMsig = plt.plot(stimMeanVec,np.ones_like(stimMeanVec),'*',visible=False,color='blue')
        self.lPMsig = plt.plot(proxMeanVec,np.ones_like(proxMeanVec),'*',visible=False,color='red')
        self.lDMsig = plt.plot(distMeanVec,np.ones_like(distMeanVec),'*',visible=False,color='green')
        
        h,p = stats.ttest_ind(self.NormBg[:,:3].flatten(),self.NormBg[:,3:].flatten())
        rax = plt.axes([0.12, 0.09, 0.09, 0.05])
        TextBox(rax, "T-test p value:",initial=str(np.round(p,6)))

        rax = plt.axes([0.025, 0.2, 0.35, 0.4], facecolor=axcolor)
        self.label = ["Stimulated spines","All spines","Mean stimulated spines","Mean proximal spines","Mean distal spines"]
        self.visibility = [False,False,False,False,False]
        self.check = CheckButtons(rax, self.label, self.visibility)
        self.check.on_clicked(self.func)

        self.axfreq = plt.axes([0.3, 0.05, 0.6, 0.03])
        self.sdist = Slider(self.axfreq, 'Threshold', np.min(self.dist)+0.01, np.max(self.dist)-0.01, valinit=self.cutoff, valstep=self.delta_dist)
        self.sdist.on_changed(self.update)
        
        rax = plt.axes([0.025, 0.65, 0.35, 0.23], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('Unnormalized', 'Normalized', 'Norm-background'), active=0,activecolor='black')
        self.radio.on_clicked(self.modefunc)
        
        saveax = plt.axes([0.025, 0.025, 0.1, 0.04])
        button = Button(saveax, 'Save', hovercolor='0.975')
        button.on_clicked(self.save)
        
        self.ax.legend()
        self.ax.set_ylabel('Unnormalized luminosity')
        self.ax.set_xlabel('Time (min)')
        plt.show()


    def func(self,label):

        """Function that toggles visibility of chosen lines"""
        yMax = 1
        yMin = 1

        if(label=="Mean stimulated spines"):
            visibilityErrbar(self.StimMean)
            ln ,_,_ = self.StimMean

            if(ln.get_visible()):

                self.StimMean.set_label('Mean stimulated spines')
                self.lSMsig[0].set_visible(True)

                if(yMax<ln.get_ydata().max()):
                    yMax = ln.get_ydata().max()

                if(yMin>ln.get_ydata().min()):
                    yMin = ln.get_ydata().min()

            else:
                self.StimMean.set_label('_nolegend_')
                self.lSMsig[0].set_visible(False)

            self.lSMsig[0].set_ydata(1.3*yMax*np.ones_like(self.lSMsig[0].get_ydata()))

        elif(label=="Mean proximal spines"):

            visibilityErrbar(self.ProxMean)
            ln ,_,_ = self.ProxMean

            if(ln.get_visible()):

                self.ProxMean.set_label('Mean proximal spines')
                self.lPMsig[0].set_visible(True)

                if(yMax<ln.get_ydata().max()):
                    yMax = ln.get_ydata().max()
                if(yMin>ln.get_ydata().min()):
                    yMin = ln.get_ydata().min()

            else:
                self.ProxMean.set_label('_nolegend_')
                self.lPMsig[0].set_visible(False)

            self.lPMsig[0].set_ydata(1.3*yMax*np.ones_like(self.lPMsig[0].get_ydata()))

        elif(label=="Mean distal spines"):

            visibilityErrbar(self.DistMean)
            ln ,_,_ = self.DistMean

            if(ln.get_visible()):

                self.DistMean.set_label('Mean distal spines')
                self.lDMsig[0].set_visible(True)

                if(yMax<ln.get_ydata().max()):
                    yMax = ln.get_ydata().max()
                if(yMin>ln.get_ydata().min()):
                    yMin = ln.get_ydata().min()

            else:
                self.DistMean.set_label('_nolegend_')
                self.lDMsig[0].set_visible(False)

            self.lDMsig[0].set_ydata(1.3*yMax*np.ones_like(self.lDMsig[0].get_ydata()))

        elif(label=="Stimulated spines"):

            for lines in self.stimLines:

                lines[0].set_visible(not lines[0].get_visible())

                if(yMax<lines[0].get_ydata().max()):
                    yMax = lines[0].get_ydata().max()

                if(yMin>lines[0].get_ydata().min()):
                    yMin = lines[0].get_ydata().min()

        elif(label=="All spines"):

            for lines in self.allLines:

                lines[0].set_visible(not lines[0].get_visible())

                if(yMax<lines[0].get_ydata().max()):
                    yMax = lines[0].get_ydata().max()

                if(yMin>lines[0].get_ydata().min()):
                    yMin = lines[0].get_ydata().min()

        yMin= 0.5*yMin
        yMax = 1.5*yMax

        self.ax.legend()
        self.ax.set_ylim(yMin,yMax)

        plt.draw()
        
    def save(self,event):

        """Function that saves the plot with the right name"""
        
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        name = self.Dir +'/'+ self.radio.value_selected +self.cellname+ '_' + str(np.round(self.sdist.val,2))+'.png'
        
        plt.savefig(name,format='png',bbox_inches=extent.expanded(1.3, 1.3))
        
    def modefunc(self,label):

        """Function that changes the data and replots the lines"""
        if(label=="Unnormalized"):
            self.stimData = self.stim
            self.SpineData = self.Unnorm
            self.compareline.set_visible(False)
            self.ax.set_ylabel('Unnormalized luminosity')

        elif(label=="Normalized"):
            self.stimData = self.stimNorm
            self.SpineData = self.Norm
            self.compareline.set_visible(True)
            self.ax.set_ylabel('Normalized luminosity')

        elif(label=='Norm-background'):
            self.stimData = self.StimNormBg
            self.SpineData = self.NormBg
            self.compareline.set_visible(True)
            self.ax.set_ylabel('Normalized luminosity - bg')

        update_errorbar(self.ProxMean,self.time,np.nanmean(self.SpineData[self.dist<self.cutoff],0)
                            ,yerr=stats.sem(self.SpineData[self.dist<self.cutoff],0))
        update_errorbar(self.DistMean,self.time,np.nanmean(self.SpineData[self.dist>self.cutoff],0)
                            ,yerr=stats.sem(self.SpineData[self.dist>self.cutoff],0))
        update_errorbar(self.StimMean,self.time, np.nanmean(self.stimData,0),
                                 yerr=stats.sem(self.stimData,0))

        self.stimMeansig = np.array([0.05>stats.ttest_ind(t,self.stimData[:,:3].flatten(),nan_policy='omit').pvalue for t in self.stimData[:,3:].T])
        self.proxMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist<self.cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<self.cutoff,3:].T])
        self.distMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist>self.cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist>self.cutoff,3:].T])

        stimMeanVec = np.array(self.time[3:]).astype(float)
        proxMeanVec = np.array(self.time[3:]).astype(float)
        distMeanVec = np.array(self.time[3:]).astype(float)

        stimMeanVec[~self.stimMeansig] = math.nan
        proxMeanVec[~self.proxMeansig] = math.nan
        distMeanVec[~self.distMeansig] = math.nan

        self.lSMsig[0].set_xdata(stimMeanVec)
        self.lPMsig[0].set_xdata(proxMeanVec)
        self.lDMsig[0].set_xdata(distMeanVec)
        
        i=0
        for l in self.stimLines:
            l[0].set_ydata(self.stimData[i,:])
            i+=1
        
        i=0
        for l in self.allLines:
            l[0].set_ydata(self.SpineData[i,:])
            i+=1

        self.ax.set_ylim([0,max(np.max(self.SpineData),np.max(self.stimData))])

        plt.draw()
        
    def update(self,val):

        """Function that updates the means based on the value of the chosen threshold"""

        cutoff = self.sdist.val

        update_errorbar(self.ProxMean,self.time,np.nanmean(self.SpineData[self.dist<cutoff],0)
                            ,yerr=stats.sem(self.SpineData[self.dist<cutoff],0))
        update_errorbar(self.DistMean,self.time,np.nanmean(self.SpineData[self.dist>cutoff],0)
                            ,yerr=stats.sem(self.SpineData[self.dist>cutoff],0))

        self.proxMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist<cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<cutoff,3:].T])
        self.distMeansig = np.array([0.05>stats.ttest_ind(t,self.SpineData[self.dist>cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist>cutoff,3:].T])

        t = np.array([stats.ttest_ind(t,self.SpineData[self.dist<self.cutoff,:3].flatten(),nan_policy='omit').pvalue for t in self.SpineData[self.dist<self.cutoff,3:].T])

        proxMeanVec = np.array(self.time[3:]).astype(float)
        distMeanVec = np.array(self.time[3:]).astype(float)
        proxMeanVec[~self.proxMeansig] = math.nan
        distMeanVec[~self.distMeansig] = math.nan

        self.lPMsig[0].set_xdata(proxMeanVec)
        self.lDMsig[0].set_xdata(distMeanVec)
        
        self.cutoff = cutoff
        self.fig.canvas.draw_idle()

def update_errorbar(errobj, x, y, xerr=None, yerr=None):

    """Function that toggles the visibility of errorbars"""
    ln, caps, bars = errobj

    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass

def visibilityErrbar(errobj):

    """Function that toggles the visibility of caps and bars of errorbars"""

    ln, caps, bars = errobj
    ln.set_visible(not ln.get_visible())

    for cap in caps:
        cap.set_visible(not cap.get_visible())
    for bar in bars:
        bar.set_visible(not bar.get_visible())

if __name__ == '__main__':
    cellType   =  'CA1_control'
    Dir        = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/NewCode/July_2019/cell_1/'
    Acceptable = ["cell_"+str(j) for j in range(1)]
    AcceptableDict = {}
    for a in Acceptable:
        AcceptableDict[a] = True
    Data,dist,bg,lenTime = GetData(Dir,AcceptableDict)
    #Wind = DataAnalWindow(Data,Dir,AcceptableDict,lenTime)
    Thresholds = [1.0,2.0,3.0]
    #Wind = MultiLineWindow(Data,Dir,Thresholds,AcceptableDict,lenTime)
    Wind =  ContourWindow(Data,dist,bg, Dir, AcceptableDict, lenTime,12)