import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,Slider,CheckButtons
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from Utility import *

import os

from SynapseFuncs import *
import torch

class SynapseClicker(Clicker):
    
    #TODO: Describe class

    def __init__(self,SynArr,tiff_Arr,frame=None):
        self.SynArr = SynArr
        self.tiff_Arr = tiff_Arr
        self.fig,self.ax = plt.subplots(figsize=(12,12))

        super().__init__(frame)

        nax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.nbutton = Button(nax, 'Next', hovercolor='0.5')
        self.nbutton.on_clicked(self.Close)

        caveax = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.cbutton = Button(caveax, 'Cancel', hovercolor='0.5')
        self.cbutton.on_clicked(self.Cancel)

class RoiCorrector(SynapseClicker):

    #TODO: Describe class

    def __init__(self, SynArr,DendArr,tiff_Arr,Simvars):

        self.Synpts = np.array([Syn.location for Syn in SynArr])
        self.Snap   = 0
        self.Channel = 0
        self.Simvars = Simvars
        self.DendArr = DendArr
        self.epsilon = 5
        self.showverts = True

        super().__init__(SynArr,tiff_Arr,Simvars.frame)
        
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
        self.labels = ['Edges','Fall Off','Dendrite','Increasing']
        self.conds = [True,True,True,True]
        self.check = CheckButtons(rax, self.labels, self.conds)
        self.check.on_clicked(self.checkfunc)
        
        # Rerun button
        rax = plt.axes([0.85, 0.3, 0.1, 0.04])
        self.rerunbutton = Button(rax, 'Rerun', hovercolor='0.5')
        self.rerunbutton.on_clicked(self.Rerun)

        #Slider to decide value of sigma in the canny edge detector
        rax = plt.axes([0.85, 0.45, 0.12, 0.03])
        self.sigslide = Slider(rax, 'sigma', 0, 10, valinit=1.5, valstep=0.5)
        self.sigslide.on_changed(self.ChangeSig)
        self.sigslide.label.set_position((0.5,1.1))
        self.sigslide.label.set_horizontalalignment('center')
        self.sigma = 1.5

        # Leniency of the ROI generator
        rax = plt.axes([0.85, 0.4, 0.12, 0.03])
        self.tolslide = Slider(rax, 'tolerance', 0, 10, valinit=3, valstep=1)
        self.tolslide.on_changed(self.ChangeTol)
        self.tolslide.label.set_position((0.5,1.1))
        self.tolslide.label.set_horizontalalignment('center')
        self.tol = 3
        
        
        self.SynapseClick = 0
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
        with open(self.Simvars.Dir + 'RoiConds.txt', 'w') as f:
            for l,c in zip(condLabels,self.conds):
                f.write(l+str(c)+'\n')
            f.write('Sigma for Canny edge: '+str(self.sigma)+'\n')
            f.write('Tolerances: '+str(self.tol)+'\n')

    def Rerun(self,event):
        
        #TODO: Describe class
        #TODO: Loading bar when rerunning
        
        self.WriteConds()
            
        for Syn in self.SynArr:
            Syn.Times  = self.Simvars.Times
            other_pts = np.round([S.location for S in self.SynArr]).astype(int)
            Syn.xpert,_,Syn.shift = FindShape(self.tiff_Arr[:,self.Channel,:,:],np.array(Syn.location),self.DendArr,
                other_pts,self.Simvars.bgmean[:,self.Channel].max(),True,self.sigma,self.conds,self.tol)

        self.RoiPlot()
        
    def InitInteractor(self,poly):
        
        #TODO: Describe class
        
        self.ax.patch.set_alpha(0.5)
        canvas = poly.figure.canvas
        x, y = zip(*poly.xy)
        
        line = Line2D(x, y,
                    marker='o', markerfacecolor='r',
                    markersize=self.epsilon,fillstyle='full',linestyle=None,linewidth=1.5,animated=True,antialiased=True)

        self.ax.add_line(line)
        
        self._ind = None  # the active vert
        
        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.CanvasList.append(canvas)
        self.PolyList.append(poly)
        self.LineList.append(line)
        
    def RoiPlot(self):
        
        #TODO: Describe class

        self.ax.cla()
        self.PolyList = []
        self.LineList = []
        self.CanvasList = []
        
        for i,Synapse in enumerate(self.SynArr):
            try:
                xplt = np.vstack([Synapse.xpert,Synapse.xpert[0]])+Synapse.shift[self.Snap][::-1]
            except:
                xplt = np.vstack([Synapse.xpert[self.Snap],Synapse.xpert[self.Snap][0]])
            poly = Polygon(xplt,fill=False, animated=True)
            self.ax.add_patch(poly)
            self.InitInteractor(poly)
            
        self.ax.imshow(self.tiff_Arr[self.Snap,self.Channel,:,:])
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
        self.background = self.CanvasList[self.SynapseClick].copy_from_bbox(self.ax.bbox)
        for p in self.PolyList:
            self.ax.draw_artist(p)
            self.ax.set_alpha(0.1)
        for l in self.LineList:
            self.ax.set_alpha(0.1)
            self.ax.draw_artist(l)
    
    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
    
        xy = np.asarray(self.PolyList[self.SpineClick].xy)
        xyt = self.PolyList[self.SpineClick].get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def OnClick(self, event):

        """Callback for mouse button presses."""

        self.SpineClick = np.argmin([np.linalg.norm(np.array((l.get_xdata(),l.get_ydata())).T-np.array([event.xdata,event.ydata]),axis=1).min() for l in self.LineList])
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
        for i,Syn in enumerate(self.SynArr):
            Syn.xpert = (np.array(self.PolyList[i].xy)[:-1] - Syn.shift[self.Snap][::-1]).tolist()
            
    def on_key_press(self, event):

        """Callback for key presses."""

        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            for l in self.LineList:
                l.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.PolyList[self.SpineClick].xy = np.delete(self.PolyList[self.SpineClick].xy,
                                         ind, axis=0)
                self.LineList[self.SpineClick].set_data(zip(*self.PolyList[self.SpineClick].xy))
        elif event.key == 'backspace':
            self.SynArr = np.delete(self.SynArr,self.SpineClick).tolist()
            self.RoiPlot()
        elif event.key == 'i':
            xys = self.PolyList[self.SpineClick].get_transform().transform(self.PolyList[self.SpineClick].xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.PolyList[self.SpineClick].xy = np.insert(
                        self.PolyList[self.SpineClick].xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.LineList[self.SpineClick].set_data(zip(*self.PolyList[self.SpineClick].xy))
                    break
        if self.LineList[self.SpineClick].stale:
            self.CanvasList[self.SpineClick].draw_idle()
    
    def getPolyXYs(self):
        return self.PolyList[self.SpineClick].xy

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

        self.PolyList[self.SpineClick].xy[self._ind] = x, y
        if self._ind == 0:
            self.PolyList[self.SpineClick].xy[-1] = x, y
        elif self._ind == len(self.PolyList[self.SpineClick].xy) - 1:
            self.PolyList[self.SpineClick].xy[0] = x, y
        self.LineList[self.SpineClick].set_data(zip(*self.PolyList[self.SpineClick].xy))

        self.CanvasList[self.SpineClick].restore_region(self.background)
        self.ax.draw_artist(self.PolyList[self.SpineClick])
        self.ax.set_alpha(0.1)
        self.ax.draw_artist(self.LineList[self.SpineClick])
        self.CanvasList[self.SpineClick].blit(self.ax.bbox)

class SynapseBuilder(SynapseClicker):

    """
    Class that defines the synapse clicking function
    """

    def __init__(self, SynArr,SingleClick,cArr,tiff_Arr,DendArr,Simvars,channel_selected,GreenChannel=False):

        super().__init__(SynArr,Simvars.z_type(tiff_Arr,axis=0),Simvars.frame)
        self.OldFlag = False
        self.RadArr = []
        self.TypeArr = []
        self.state = 0
        self.SynNum = 0
        self.cArr  = cArr
        self.channel_selected = channel_selected

        self.ax.set_title('Click the synapses')
        self.ax.imshow(tiff_Arr[0,:,:])
        
        Dir   = Simvars.Dir
        
        try:
            OldSynArr = ReadSynDict(Dir,0,Simvars.Unit,Simvars.Mode)
            pts = np.array([S.location for S in OldSynArr])
            flag = np.array([1 if S.type =="Stim" else 0 for S in OldSynArr])
            score = np.ones_like(flag)
            ROIs = [S.xpert for S in OldSynArr]
            self.ptI  = self.ptInteractor(pts,score,0.5,self.fig,self.ax,flag,ROIs)
        except:
            pts = []
            Lines = np.array([])
            score = np.array([])
            flag = np.array([])
            ROIs = []
            self.ptI  = self.ptInteractor(pts,score,0.5,self.fig,self.ax)

        try:
            for d in DendArr:
                self.ax.plot(d[:,0],d[:,1],'x-r',markersize=0.5)
        except:
            self.ax.plot(DendArr[:,0],DendArr[:,1],'x-r',markersize=0.5)
        
        nax = plt.axes([0.7, 0.025, 0.1, 0.04])
        self.nnbutton = Button(nax, 'Run NN', hovercolor='0.5')
        self.nnbutton.on_clicked(self.NNButton(Simvars,DendArr,tiff_Arr))


        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        plt.show()

    def NNButton(self,Simvars,DendArr,tiff_Arr):

        #TODO: Describe function

        def clicked(event):
            OldPts    = self.ptI.pts
            OldScores = self.ptI.score
            OldFlag   = self.ptI.flag
            OldROI    = self.ptI.ROIs

            pPoints,score = self.RunNN(Simvars,DendArr,tiff_Arr)
            if(len(score)>0):
                if(self.ptI.just_deleted):
                    pts = pPoints
                else:
                    pts   = np.vstack([OldPts,pPoints])
                ROIs  = OldROI + [[]]*len(score)
                scores = np.append(OldScores,score)
                flag  = np.append(OldFlag,np.zeros_like(score))
            else:
                pts = OldPts
                ROIs = OldROI
                scores = OldScores
                flag = OldFlag

            self.ax.cla()
            self.ax.set_title('Click the synapses')
            self.ax.imshow(tiff_Arr[0,:,:])
            try:
                for d in DendArr:
                    self.ax.plot(d[:,0],d[:,1],'x-r',markersize=0.5)
            except:
                self.ax.plot(DendArr[:,0],DendArr[:,1],'x-r',markersize=0.5)
            self.ptI = self.ptInteractor(pts,scores,0.5,self.fig,self.ax,flag,ROIs)
            if not hasattr(self, 'sdist'): 
                axfreq = plt.axes([0.35, 0.9, 0.3, 0.03])
                self.sdist = Slider(axfreq, 'ML Confidence', 0, 1, valinit=0.5, valstep=0.05)
                self.sdist.on_changed(self.ptI.ChangeSig)
            plt.draw()
        return clicked

    def OnClick(self,event):
        if(event.ydata>1):
            self.ptI.OnClick(event)
            plt.draw()
    def RunNN(self,Simvars,DendArr,tiff_Arr):

        #TODO: Describe function

        Box_size = 32
        model = torch.load(Simvars.model,map_location=torch.device('cpu'))
        model.eval()
        BoxsList = []
        ScoreList = []
        offset = [0]#,Box_size//4,Box_size//2]
        Training_x = []
        Training_y = []
        for d in DendArr:
            Training_x.append([max(min(d[:,1])-Box_size,0),min(max(d[:,1])+Box_size,tiff_Arr.shape[-1]-Box_size)])
            Training_y.append([max(min(d[:,0])-Box_size,0),min(max(d[:,0])+Box_size,tiff_Arr.shape[-2]-Box_size)])

        Training_x = np.array(Training_x)
        Training_x = [Training_x[0,:].min(),Training_x.max()]
        Training_y = np.array(Training_y)
        Training_y = [Training_y[0,:].min(),Training_y.max()]
   
        Y = np.arange(Training_y[0],Training_y[1],Box_size)
        X = np.arange(Training_x[0],Training_x[1],Box_size)
        im = Simvars.z_type(tiff_Arr,axis=0)[None,:,:]
        im = np.repeat(im,3,axis=0)
        im = data_transforms['val'](np.moveaxis(im,0,-1).astype(np.uint8))[None,:,:,:]
        for o in offset:
            for x in X:
                for y in Y:
                    y = y+o
                    img = im[:,:,x:x+Box_size,y:y+Box_size]
                    testOut = model(img)    
                    boxs = testOut[0]['boxes'].detach().numpy()
                    boxs[:,(0,2)], boxs[:,(1,3)] = boxs[:,(0,2)]+y,boxs[:,(1,3)] + x
                    scores = testOut[0]['scores'].detach().numpy().tolist()
                    BoxsList = BoxsList + boxs.tolist()
                    ScoreList = ScoreList + scores
        sBoxsList = [x for _, x in sorted(zip(ScoreList, BoxsList))]
        sScoreList = sorted(ScoreList)
        sScoreList = np.array(sScoreList[::-1])
        sBoxsList = np.array(sBoxsList[::-1])

        i = 0
        tBoxsList = np.copy(sBoxsList)
        tScoreList = np.copy(sScoreList)
        while i < len(tBoxsList):
            poplist = []
            for j,b in enumerate(tBoxsList):
                if(iou(tBoxsList[i],b)>0 and iou(tBoxsList[i],b)<1): poplist.append(j)
            tBoxsList = np.delete(tBoxsList,poplist,axis=0)
            tScoreList = np.delete(tScoreList,poplist)
            i = i+1          
        pPoints = []
        score = tScoreList
        for b in tBoxsList:
            pPoints.append([(b[0]+b[2])/2, (b[1]+b[3])/2])
        pPoints = np.array(pPoints)
        return pPoints,score

    # TODO: Describe the functions below
    def CheckContour(self,event):
        if(self.cArr[np.round(event.ydata).astype(int),np.round(event.xdata).astype(int)]):
            print('Clicked a contour!')
            return False
        return True

    def CorrectPath(self,event):
        self.OldFlag = True
        plt.close()

    class ptInteractor():
        
         #TODO: Describe class

        def __init__(self,pts,score,sigma,fig,ax,flag=None,ROIs=None):
            self.Lines = []
            self.sigma = sigma
            self.score = score
            self.ax    = ax
            self.fig   = fig
            self.pts   = pts
            self.just_deleted = False
            if(len(pts)==0):
                self.just_deleted = True
            if(flag is None):
                self.flag  = np.zeros_like(score)
            else:
                self.flag = flag
            if(ROIs is None):
                self.ROIs = [[]]*len(score)
            else:
                self.ROIs = ROIs
            for p in pts:
                self.Lines.append(self.ax.plot(p[0],p[1],'xr'))
            self.Lines = np.array(self.Lines).squeeze()
            if(self.Lines.ndim==0): self.Lines = [self.Lines]
            for p,s in zip(self.Lines,self.score):
                if(s<self.sigma):
                    p.set_visible(False)

            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.showpts = True
            
        def on_key_press(self, event):

            #TODO: Describe function

            if not event.inaxes:
                return
            if event.key == 't':
                self.showpts = not self.showpts
                if(self.showpts):
                    for l in self.Lines:
                        if(l.get_visible()):
                            l.set_visible(False)
                else:
                    for p,s in zip(self.Lines,self.score):
                        if(s>self.sigma):
                            p.set_visible(True)
            elif event.key == 'i':
                if(self.just_deleted):
                    self.pts = np.array([[event.xdata,event.ydata]])
                    self.just_deleted = False
                else:
                    self.pts = np.vstack([self.pts,[event.xdata,event.ydata]])
                self.score = np.append(self.score,1)
                self.Lines = np.append(self.Lines,self.ax.plot(event.xdata,event.ydata,'xr',visible=True))
                self.flag  = np.append(self.flag,0)
                self.ROIs.append([])
            elif event.key == 'd':
                if((np.linalg.norm(self.pts-[event.xdata,event.ydata],axis=1)<15).any()):
                    id_x = np.argmin(np.linalg.norm(self.pts-[event.xdata,event.ydata],axis=1))
                    self.Lines[id_x].remove()
                    self.Lines = np.delete(self.Lines,id_x)
                    self.score = np.delete(self.score,id_x)
                    self.flag  = np.delete(self.flag,id_x)
                    self.pts   = np.delete(self.pts,id_x,axis=0)
                    self.ROIs.pop(id_x)
            elif event.key == 'D':
                for l in self.Lines:
                    l.remove()
                self.pts = []
                self.Lines = np.array([])
                self.score = np.array([])
                self.flag = np.array([])
                self.ROIs = []
                self.just_deleted = True
            elif event.key == 'a':
                if((np.linalg.norm(self.pts-[event.xdata,event.ydata],axis=1)<15).any()):
                    id_x = np.argmin(np.linalg.norm(self.pts-[event.xdata,event.ydata],axis=1))
                    if(self.flag[id_x] == 1):
                        self.Lines[id_x].set_color('red')
                        self.flag[id_x] = 0
                    else:
                        self.Lines[id_x].set_color('yellow')
                        self.flag[id_x] = 1
            plt.draw()

        def OnClick(self, event):
            if(self.just_deleted):
                self.pts = np.array([[event.xdata,event.ydata]])
                self.just_deleted = False
            else:
                self.pts = np.vstack([self.pts,[event.xdata,event.ydata]])
            self.score = np.append(self.score,1)
            self.Lines = np.append(self.Lines,self.ax.plot(event.xdata,event.ydata,'xr',visible=True))
            self.flag  = np.append(self.flag,0)
            self.ROIs.append([])    

        def ChangeSig(self,val):

            #TODO: Describe function

            self.sigma = val
            for p,s in zip(self.Lines,self.score):
                if(s<self.sigma):
                    p.set_visible(False)
                else:
                    p.set_visible(True)