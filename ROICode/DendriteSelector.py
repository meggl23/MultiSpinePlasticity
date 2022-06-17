import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button,Slider,CheckButtons
from matplotlib.patches import Polygon
from skimage.draw import polygon,ellipse

from skimage.filters import threshold_triangle, threshold_mean

import PathFinding as PF
from SynapseFuncs import FindShape,ShapeArea

import sys

import os

from Utility import *
from LineInteractor import *

from skimage import feature

Modes = ["Create","Update"]

class DendriteMeasure:

    def __init__(self,tiff_Arr,SimVars):
        self.tiff_Arr = tiff_Arr
        self.channel_selected = 0
        self.Dir      = SimVars.Dir
        self.bg      = SimVars.bgmean
        self.Unit    = SimVars.Unit
        self.DoneFlag = False
        self.NextFlag = True
        self.Soma     = []
        self.SomaBord = []
        self.isframe    = False
        self.mode = Modes[0]
        self.ShowStats = True
        if(SimVars.frame == None):
            self.isframe    = True
        self.frame = SimVars.frame
        
    def Automatic(self,SimVars):

        """
        Function that organises the gui for the interaction
        with the dendrite clicking tool.

        """

        self.fig,self.ax = plt.subplots(figsize=(10,10))

        oldDend,somas,dend_names = LoadDend(self.Dir)
        cc = [x for x in range(1,self.tiff_Arr.shape[0]+1)]
        axcolor = 'lightgoldenrodyellow'
        axchannel = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor=axcolor)
        self.schannel = Slider(axchannel, 'Channel',valmin=cc[0],valmax=cc[-1],valinit=cc[0],valstep=1)
            
        self.schannel.on_changed(self.ChangeChannel)

        self.ax.set_title('Click the start and ends of the dendrite')

        oldDend,somas,dend_names = LoadDend(self.Dir)
        cc = [x for x in range(1,self.tiff_Arr.shape[0]+1)]

        if(len(oldDend)>0):
            self.mode = Modes[1]
            nax = plt.axes([0.65, 0.025, 0.1, 0.04])
            kbutton = Button(nax, 'Keep Old?', hovercolor='0.5')
            kbutton.on_clicked(self.CorrectPath)
            nax = plt.axes([0.55, 0.025, 0.1, 0.04])
            clbutton = Button(nax, 'Clear', hovercolor='0.5')
            clbutton.on_clicked(self.Clear)
            self.SomaBord = somas
            soma_arrs = self.getSomaPoly()
            for soma_arr in soma_arrs[:]:
                self.ax.plot(soma_arr[:,0],soma_arr[:,1],'or-',linewidth=2)
            length = []
            self.p = []
            for ind,d in enumerate(oldDend):
                line = Polygon(d, closed=False,fill=False,animated=True)
                self.ax.add_patch(line)
                self.p.append(LineInteractor(self.ax, line))
                length.append(sum([np.linalg.norm(d1-d2) for d1,d2 in zip(d[:-1],d[1:])])*self.Unit)
            rax = plt.axes([0.25, 0.025, 0.1, 0.04], facecolor=axcolor)
            self.label = ['ShowStats']
            self.cond = [False]
            self.check = CheckButtons(rax, self.label, self.cond)
            self.check.on_clicked(self.checkfunc)
        else:        
            nax = plt.axes([0.55, 0.025, 0.1, 0.04])
            nbutton = Button(nax, 'Next', hovercolor='0.5')
            nbutton.on_clicked(self.Next)
        self.ax.imshow(self.tiff_Arr[self.channel_selected,:,:])
        DendArr = []
        
        caveax = plt.axes([0.45, 0.025, 0.1, 0.04])
        cbutton = Button(caveax, 'Cancel', hovercolor='0.5')
        cbutton.on_clicked(self.Cancel)
        self.DendBuild = DendriteBuilder(DendArr,self.fig,self.ax,SimVars.frame,self.channel_selected,True,self.mode)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        if(self.DoneFlag):
            DendArr = []
            for od in (oldDend):
                #DendArr.append(od)
                DendArr.append(PF.GetAllpointsonPath(od))
            return DendArr,length,self.SomaBord,self.channel_selected

        DendArr = self.DendBuild.DendArr
        DendArr = np.round(DendArr).astype(int)
    
        
        self.Soma = np.round(self.DendBuild.Soma).astype(int)

        if not(SimVars.frame==None):
            SimVars.frame.v.set(SimVars.frame.v.get()+"Finding the paths through the dendrite \n")
            SimVars.frame.master.update()

        DendArr_m = []
        if(len(self.Soma)==0):
            start = np.array(np.round([DendArr[0][1],DendArr[0][0]])).astype(int)
            end = np.array(np.round([DendArr[1][1],DendArr[1][0]])).astype(int)

            Mesh,Ds,ls,fp,sp = self.GenMesh(start,end,self.Unit)
            i=0
            sp = self.PlotDend(Mesh,Ds,'last',fp,sp)
            np.save(self.Dir+"Dendrite0.npy", sp)
            DendArr_m.append(PF.GetAllpointsonPath(sp))
            length = PF.PathLength(sp,self.Unit)
            return DendArr_m,length,self.SomaBord,self.channel_selected
        else:
            self.SomaBord,area,SomaDrift = FindShape(self.tiff_Arr[self.channel_selected,:,:],self.Soma,np.array(DendArr),[],self.bg[:,0].max(axis=0),False,10,[True,True,False,False])
            SC = SomaCorrector(self.Soma,self.SomaBord,self.tiff_Arr,SimVars)
            self.SomaBord,area = SC.SomaBord,SC.area
            DendNum = 0
            length_m  = []
            for kk,d in enumerate(DendArr):
                if(not(self.Soma==d).all()):
                    if not(self.frame==None):
                        self.frame.v.set(self.frame.v.get()+"Working out the dendrite "+str(kk)+ "! \n")
                        self.frame.progress['value'] = (kk/len(DendArr)*100)
                        self.frame.master.update()
                    start = d
                    DendNum+=1
                    rr,cc = polygon(np.array(self.SomaBord)[:,1],np.array(self.SomaBord)[:,0], self.tiff_Arr[self.channel_selected,:,:].shape)
                    Mesh,Ds,ls,fp,sp = self.GenMesh(np.array(np.round([start[1],start[0]])).astype(int),np.column_stack((rr,cc)),self.Unit)
                    i=0
                    sp = self.PlotDend(Mesh,Ds,'last',fp,sp)
                    Ds = PF.GetAllpointsonPath(sp)
                    np.save(self.Dir+"Dendrite"+str(kk)+".npy", sp)
                    length = PF.PathLength(sp,self.Unit)
                    DendArr_m.append(Ds)
            if (len(self.SomaBord)>0):
                np.save(self.Dir+"Soma_1.npy", self.SomaBord)
                
        return np.squeeze(DendArr_m),length_m,self.SomaBord,self.channel_selected

    def checkfunc(self,label):
        
        #TODO: Describe function
        
        self.cond[0] =  not self.cond[0]
        
        self.ShowStats = self.cond[0]

    def GenMesh(self,start,end,scale):
        Mesh = []
        Ds = []
        ls = []
        i=1
        length=1
        lengthOld=0
        Evals = 0
        thresh = threshold_mean(self.tiff_Arr)
        while length>0:
            binary = self.tiff_Arr[self.channel_selected,:,:] > i*thresh
            Mesh.append(np.where(binary == True, 1, 0))
            DendArr,length = PF.FindSoma(Mesh[-1],start,end,self.Unit)
            DendArr = np.array(DendArr)
            Ds.append(DendArr)
            ls.append(length)
            # TODO - add a better way to update the threshold
            if(length==lengthOld):
                MultFact = 2
            else:
                MultFact = 1.5
            i=i*MultFact
            lengthOld = length
            Evals+=1
        Ds.pop()
        ls.pop()
        Ds.reverse()
        ls.reverse()
        Mesh.pop()
        Mesh.reverse()

        sp,fp = PF.SecondOrdersmoothening(np.asarray(Ds[0]),np.sqrt(2)/scale)
        return Mesh[0],Ds[0],ls[0],fp,sp
    def CorrectPath(self,event):
        self.DoneFlag = True
        plt.close()


    def Back(self,event):
        self.NextFlag=False
        plt.close()

    def Cancel(self,event):

        """ Function for plot buttons to cancel the program"""
        if(self.isframe):
            sys.exit()
        else:
            plt.close()

    def Clear(self,event):
        for p1 in self.p:
            p1.clear()
        self.ax.cla()
        self.ax.imshow(self.tiff_Arr[self.channel_selected,:,:])
        nax = plt.axes([0.65, 0.025, 0.1, 0.04])
        self.nbutton = Button(nax, 'Next', hovercolor='0.5')
        self.nbutton.on_clicked(self.Next)
        self.mode = Modes[0]
        self.DendBuild.mode = Modes[0]

        plt.draw()

    def getSomaPoly(self):
        soma_polys = np.asarray(self.SomaBord)
        for soma_poly in soma_polys[:]:
            if not (soma_poly[0] == soma_poly[-1]).all():
                np.append(soma_poly,[soma_poly[0]])
        return soma_polys

    def Next(self,event):
        # self.channel_selected=0
        plt.close()

    def ChangeChannel(self,val):
            self.channel_selected = val - 1
            # print(" viewing channel = ",self.channel_selected )
            self.ax.imshow(self.tiff_Arr[self.channel_selected,:,:])
            plt.draw()


    def PlotDend(self,mesh,DendArr,LastFirst='none',fp=None,sp=None):

        """
        Function to plot resulting dendrites from the automatic path
        """

        fig,ax = plt.subplots(1,2,figsize=(20,10))
        ax[0].imshow(mesh)
        soma_arr = self.getSomaPoly()
        if len(soma_arr.shape) == 2:
            ax[1].plot(soma_arr[:,0],soma_arr[:,1],'x-r')
        try:
            ax[1].imshow(self.tiff_Arr[self.channel_selected,:,:])
            
        except:
            ax[1].imshow(self.tiff_Arr[self.channel_selected,:,:])
        if sp is not None:
            line = Polygon(sp, closed=False,fill=False,animated=True)
            ax[1].add_patch(line)
            p1 = LineInteractor(ax[1], line)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
            sp = p1.getPolyXYs()
            sp = sp.astype(int)
            return sp
        else:
            try:
                ax[1].scatter(DendArr[:,0],DendArr[:,1])
            except:
                ax[1].scatter(DendArr[0],DendArr[1])
            return DendArr
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def GetAdaptiveWidths(self,all_ps,sigma):
        mean_img = self.tiff_Arr[self.channel_selected,:,:] > threshold_mean(self.tiff_Arr[self.channel_selected,:,:])

        width_arr,degrees = getWidthnew(mean_img,all_ps,sigma)
        
        window_size = int(1 // self.Unit)
        avg_width = moving_average(width_arr,window_size)
        max_width = width_arr.max()
        max_width_arr = np.ones(width_arr.shape)*max_width
        rr1,cc1 = plotWidth(self.Dir,"max_width",self.tiff_Arr[self.channel_selected,:,:],all_ps,max_width_arr,degrees)
        rr2,cc2 = plotWidth(self.Dir,"adaptive_width",self.tiff_Arr[self.channel_selected,:,:],all_ps,avg_width,degrees)

        return GetDendriticStats(self.tiff_Arr,rr2,cc2),GetDendriticStats(self.tiff_Arr,rr1,cc1),np.column_stack((all_ps,avg_width,width_arr,degrees))


class DendriteBuilder(Clicker):

    """
    Class that defines the dendrite clicking function
    """

    def __init__(self,DendArr, fig,ax,frame,channel_selected,Auto=True,mode=Modes[0]):

        self.DendArr = DendArr
        self.fig = fig
        self.ax    = ax
        self.Auto  = Auto
        self.Soma     = []
        self.channel_selected = channel_selected
        self.mode = mode
        super().__init__(frame)

    def OnClick(self, event):

        """
        Function that does the clicking for the dendrite.
        Shift click for the soma - then all paths will go there
        Otherwise click twice for each side of the dendrite and 
        the path between the two will be found


        Right click for deletion

        If manual mode is chosen then the full dendrite is selected manually

        """

        if (event.ydata > 1 and self.mode == Modes[0]):
            if(event.button==MouseButton.LEFT):
                if(self.fig.canvas.cursor().shape() == 0):
                    self.DendArr.append([event.xdata,event.ydata])
                    if(self.Auto):
                        Col = 'xb'
                    else:
                        Col = 'x-r'
                    if(self.key=='shift'):
                        Col = 'xg'
                        self.Soma = [event.xdata,event.ydata]
                    if(self.Auto):    
                        self.ax.plot(np.asarray(self.DendArr)[-1,0],np.asarray(self.DendArr)[-1,1],Col,mew=2,ms=8)
                    else:
                        self.ax.plot(np.asarray(self.DendArr)[:,0],np.asarray(self.DendArr)[:,1],Col,mew=2,ms=8)
                    plt.draw()
            else:
                self.DendArr.pop()
                self.ax.lines[-1].remove()
                self.Soma=[]
                plt.draw()

class SomaCorrector(Clicker):

    #TODO: Describe class

    def __init__(self, Soma,SomaBord,tiff_Arr,Simvars):

        self.Snap   = 0
        self.Channel = 0
        self.Simvars = Simvars
        self.tiff_Arr = tiff_Arr
        self.SomaBord = SomaBord
        self.Soma    = Soma
        self.epsilon = 5
        self.showverts = True
        self.fig,self.ax = plt.subplots(figsize=(12,12))

        super().__init__(Simvars.frame)
        
        #Sliders to change channel and snapshot
        axfreq = plt.axes([0.35, 0.93, 0.3, 0.03])
        self.sdist = Slider(axfreq, 'Snapshot', 1, Simvars.Snapshots, valinit=1, valstep=1)
        self.sdist.on_changed(self.ChangeSnap)
        
        axfreq = plt.axes([0.35, 0.9, 0.3, 0.03])
        self.s2dist = Slider(axfreq, 'Channel', 1, Simvars.Channels, valinit=1, valstep=1)
        self.s2dist.on_changed(self.ChangeChannel)
    
        #Buttons to decide which conditions to run in ROI generation
        axcolor = 'lightgoldenrodyellow'
        rax = plt.axes([0.85, 0.5, 0.12, 0.2], facecolor=axcolor)
        self.labels = ['Edges','Fall Off']
        self.conds = [True,True]
        self.check = CheckButtons(rax, self.labels, self.conds)
        self.check.on_clicked(self.checkfunc)
        
        # Rerun button
        rax = plt.axes([0.85, 0.3, 0.1, 0.04])
        self.rerunbutton = Button(rax, 'Rerun', hovercolor='0.5')
        self.rerunbutton.on_clicked(self.Rerun)

        #Slider to decide value of sigma in the canny edge detector
        rax = plt.axes([0.85, 0.45, 0.12, 0.03])
        self.sigslide = Slider(rax, 'sigma', 0, 10, valinit=5, valstep=0.5)
        self.sigslide.on_changed(self.ChangeSig)
        self.sigslide.label.set_position((0.5,1.1))
        self.sigslide.label.set_horizontalalignment('center')
        self.sigma = 5

        # Leniency of the ROI generator
        rax = plt.axes([0.85, 0.4, 0.12, 0.03])
        self.tolslide = Slider(rax, 'tolerance', 0, 10, valinit=3, valstep=1)
        self.tolslide.on_changed(self.ChangeTol)
        self.tolslide.label.set_position((0.5,1.1))
        self.tolslide.label.set_horizontalalignment('center')
        self.tol = 3
        
        nax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.nbutton = Button(nax, 'Next', hovercolor='0.5')
        self.nbutton.on_clicked(self.Close)

        caveax = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.cbutton = Button(caveax, 'Cancel', hovercolor='0.5')
        self.cbutton.on_clicked(self.Cancel)
        
        self.RoiPlot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def checkfunc(self,label):
        
        #TODO: Describe function
        
        index = self.labels.index(label)
        self.conds[index] =  not self.conds[index]
    
    def WriteConds(self):
        
        condLabels = ['Contour: ','Fall Off: ','Dendrite: ','Increasing: ']
        with open(self.Simvars.Dir + 'SomaConds.txt', 'w') as f:
            for l,c in zip(condLabels,self.conds):
                f.write(l+str(c)+'\n')
            f.write('Sigma for Canny edge: '+str(self.sigma)+'\n')
            f.write('Tolerances: '+str(self.tol)+'\n')

    def Rerun(self,event):
        
        #TODO: Describe class
        
        self.WriteConds()
        
        self.SomaBord,self.area,SomaDrift = FindShape(self.tiff_Arr[self.Channel,:,:],self.Soma,[],[],self.Simvars.bgmean[:,self.Channel].max(),False,self.sigma,self.conds+[False,False],self.tol)

        self.RoiPlot()
        
    def InitInteractor(self):
        
        #TODO: Describe class
        
        self.ax.patch.set_alpha(0.5)
        self.canvas = self.poly.figure.canvas
        x, y = zip(*self.poly.xy)
        
        self.line = Line2D(x, y,
                    marker='o', markerfacecolor='r',
                    markersize=self.epsilon,fillstyle='full',linestyle=None,linewidth=1.5,animated=True,antialiased=True)

        self.ax.add_line(self.line)
        
        self._ind = None  # the active vert
        
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        
    def RoiPlot(self):
        
        #TODO: Describe class

        self.ax.cla()
        
        self.poly = Polygon(self.SomaBord,fill=False, animated=True)
        self.ax.add_patch(self.poly)
        self.InitInteractor()
            
        self.ax.imshow(self.tiff_Arr[self.Channel,:,:])
        self.key = None
        plt.draw()


    #TODO: Description of the following functions
    def ChangeSnap(self,val):
        self.ax.cla()
        self.Snap = val-1
        self.RoiPlot()
        
    def ChangeChannel(self,val):
        self.ax.cla()
        self.Channel = val-1
        self.RoiPlot()
        
    def ChangeSig(self,val):
        self.sigma = val

    def ChangeTol(self,val):
        self.tol = val
        
    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.set_alpha(0.1)
        self.ax.draw_artist(self.line)
    
    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
    
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def OnClick(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None
        self.SomaBord = self.poly.xy
        self.area = ShapeArea(self.Soma,self.SomaBord)*self.Simvars.Unit**2
            
    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()
    
    def getPolyXYs(self):
        return self.poly.xy

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.set_alpha(0.1)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
def LoadDend(Dir):

    """
    Input:
            Dir (String)   : Super directory we are looking at
    Output:
            DendArr  (numpy array) : location of dendrite
        
    Function:
            Read npy files to obtain dendrites
    """
    DendArr = []
    DendArr_names = []
    soma = []
    for x in os.listdir(Dir):
        if('Dendrite' in x):
            DendArr.append(np.load(Dir+x))
            num = x.split("Dendrite")[1].split(".npy")[0]
            DendArr_names.append(num)
        elif("Soma" in x) and x.endswith(".npy"):
            soma.append(np.load(Dir+x))

    return DendArr,soma,DendArr_names


def SearchWidth(img,edges,p,degree,width,count):
    rr, cc = ellipse(p[1], p[0], width, 1, rotation=degree,shape=img.shape)
    iix = np.where(edges[rr,cc]  == True)[0]

    if(iix.shape[0] > 1 or count > 30):
        return width
    next_width = width*1.2
    count += 1
    return SearchWidth(img,edges,p,degree,next_width,count)

def getWidthnew(img,all_ps,sigma):
    edges1 = feature.canny(img,sigma=sigma)
    i_img = np.zeros(img.shape)
    width_arr = np.zeros(all_ps.shape[0])
    degrees = np.zeros(all_ps.shape[0])
    for dxd,d in enumerate(all_ps[1:]):
        u_vector = GetPerpendicularVector(all_ps[dxd],d)
        if angle_between2(np.array([0,1]),u_vector) > 180:
                degrees[dxd] =  -1*angle_between(np.array([0,1]),u_vector)
        else:
            degrees[dxd] =  angle_between(np.array([0,1]),u_vector)
        starting_width = 1
        width_arr[dxd] = SearchWidth(img,edges1,all_ps[dxd],degrees[dxd],starting_width,0)

        rr, cc = ellipse(all_ps[dxd][1], all_ps[dxd][0], width_arr[dxd], 2, rotation=degrees[dxd],shape=img.shape)

        i_img[rr,cc] = edges1[rr,cc]
    return width_arr,degrees


def plotWidth(Dir,fname,img,points,widths,degrees):
    fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(20, 12), sharex=True, sharey=True)
    null_img = np.zeros(img.shape)
    ax0.imshow(img)
    for pdx,p in enumerate(points[:]):
        rr,cc = ellipse(p[1],p[0],widths[pdx],2,rotation=degrees[pdx],shape=img.shape)
        null_img[rr,cc] = 1
    combined = np.multiply(img,null_img)
    rr,cc = np.nonzero(combined)
    ax1.imshow(combined)
    plt.savefig(Dir+fname+".png")
    plt.savefig(Dir+fname+".eps")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    return [rr,cc]


def GetDendriticStats(tiffArr,rr,cc):
    f_data = tiffArr[:,rr,cc]
    stats = {}
    stats["mean_arr"] = np.mean(f_data,axis=1).tolist()
    stats["min"]   = np.min(f_data,axis=1).tolist()
    stats["max"]  = np.max(f_data,axis=1).tolist()
    stats["RawIntDen"] = np.sum(f_data,axis=1).tolist()
    return stats