def zStackManual(tiff_Arr,Dir,FileNames):

    if(os.path.isfile(Dir+".npy")==False):
        StackCutFull = np.zeros([len(tiff_Arr),2])

        for i in range(len(tiff_Arr)-1):
            StackCut = []
            zStackManualPlot(tiff_Arr[i],tiff_Arr[i+1],StackCut,FileNames[i],FileNames[i+1])
            print(StackCut)
            for j in range(i+1):
                cut0_l = list(range(0,StackCut[0][0]))
                cut0_u = list(range(len(tiff_Arr[j])-1,len(tiff_Arr[j])-1-(len(tiff_Arr[i])-1-StackCut[0][1]),-1))
                StackCutFull[j,0] = StackCutFull[j,0] + len(cut0_l)
                StackCutFull[j,1] = StackCutFull[j,1] + len(cut0_u)
                tiff_Arr[j] = np.delete(tiff_Arr[j],cut0_u,axis=0) 
                tiff_Arr[j] = np.delete(tiff_Arr[j],cut0_l,axis=0) 
            cut1_l = list(range(0,StackCut[1][0]))
            cut1_u = list(range(len(tiff_Arr[i+1])-1,StackCut[1][1],-1))
            StackCutFull[i+1,0] = StackCutFull[i+1,0] + len(cut1_l)
            StackCutFull[i+1,1] = StackCutFull[i+1,1] + len(cut1_u)
            tiff_Arr[i+1] = np.delete(tiff_Arr[i+1],cut1_u,axis=0) 
            tiff_Arr[i+1] = np.delete(tiff_Arr[i+1],cut1_l,axis=0) 
        np.save('Dir', StackCutFull)
    else:
        StackCutFull = np.load(Dir+".npy")
        for i,s in enumerate(StackCutFull):
            cut_l = list(range(0,int(s[0])))
            cut_u = list(range(len(tiff_Arr[i])-int(s[1]),len(tiff_Arr[i])))
            tiff_Arr[i] = np.delete(tiff_Arr[i],cut_u,axis=0)
            tiff_Arr[i] = np.delete(tiff_Arr[i],cut_l,axis=0)
    
    tiff_Arr = np.asarray(tiff_Arr)
    return np.moveaxis(tiff_Arr,[0],[-1])

class zStackManualPlot:
    def __init__(self,tiff_Arr,tiff_Arr1,StackCut,Title1,Title2):
        self.tiff_Arr = tiff_Arr
        self.tiff_Arr1 = tiff_Arr1
        self.StackCut  = StackCut
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,figsize=(16,8))
        self.ax1.contourf(self.tiff_Arr[0,:,:])
        self.ax2.contourf(self.tiff_Arr1[1,:,:])
        self.ax1.set_title(Title1)
        self.ax2.set_title(Title2)
        axfreq = plt.axes([0.15, 0.93, 0.3, 0.03])
        axfreq2 = plt.axes([0.57, 0.93, 0.3, 0.03])
        self.sdist = Slider(axfreq, 'Stack', 1, tiff_Arr.shape[0], valinit=1, valstep=1)
        self.sdist.on_changed(self.ChangeLeftStack)
        self.sdist2 = Slider(axfreq2, 'Stack', 1, tiff_Arr1.shape[0], valinit=1, valstep=1)
        self.sdist2.on_changed(self.ChangeRightStack)
        axbox1 = plt.axes([0.23, 0.03, 0.04, 0.03])
        axbox2 = plt.axes([0.38, 0.03, 0.04, 0.03])
        axbox3 = plt.axes([0.65, 0.03, 0.04, 0.03])
        axbox4 = plt.axes([0.8, 0.03, 0.04, 0.03])

        self.min_1 = 1
        self.max_1 = len(tiff_Arr)
        self.min_2 = 1
        self.max_2 = len(tiff_Arr1)
        self.text_box_m1 = TextBox(axbox1, 'Min slice', initial=str(self.min_1),label_pad=0.2)
        self.text_box_m1.on_submit(self.submit1)
        self.text_box_M1 = TextBox(axbox2, 'Max slice', initial=str(self.max_1),label_pad=0.2)
        self.text_box_M1.on_submit(self.submit2)
        self.text_box_m2 = TextBox(axbox3, 'Min slice', initial=str(self.min_2),label_pad=0.2)
        self.text_box_m2.on_submit(self.submit3)
        self.text_box_M2 = TextBox(axbox4, 'Max slice', initial=str(self.max_2),label_pad=0.2)
        self.text_box_M2.on_submit(self.submit4)


        saveax = plt.axes([0.9, 0.025, 0.1, 0.04])
        self.sbutton = Button(saveax, "Next")
        self.sbutton.on_clicked(Next)

        plt.show()

        return None
        
    def ChangeLeftStack(self,val):

        stack = int(self.sdist.val)
        self.ax1.clear()
        self.ax1.contourf(self.tiff_Arr[stack-1,:,:])
        self.fig.canvas.draw_idle()

        return None

    def ChangeRightStack(self,val):

        stack = int(self.sdist2.val)
        self.ax2.clear()
        self.ax2.contourf(self.tiff_Arr1[stack-1,:,:])
        self.fig.canvas.draw_idle()

        return None


    def next(self,event):
        """Function that saves the contour plot with the right name"""

        self.StackCut.append([self.min_1-1,self.max_1-1])
        self.StackCut.append([self.min_2-1,self.max_2-1])

        plt.close()

    def submit1(self,val):
        self.min_1 = int(val)

    def submit2(self,val):
        self.max_1 = int(val)

    def submit3(self,val):
        self.min_2 = int(val)

    def submit4(self,val):
        self.max_2 = int(val)


def ConcatZstack(T,MinDirCum,nSnaps,DendArr):
    
    """
    Input:
            tiff_Arr (np.array of doubles)  : Pixel values of all the tiff files
            MinDirCum (np.array)            : The integer values that the array must be shifted
            nSnaps                          : Number of snapshots

    Output:
            Tnew (np.array of doubles)  : Pixel values of all the tiff files, with extra
                                          stacks removed

    Function:
            Rearranges the chosen z-stack array to form a consisten array.
    """

    i=0
    j=1

    lim = np.max(abs(MinDirCum))+1
    lim2 = 40

    t0 =T[i][:,lim+MinDirCum[0,i]:-lim+MinDirCum[0,i],lim+MinDirCum[1,i]:-lim+MinDirCum[1,i]]
    t1 =T[j][:,lim+MinDirCum[0,j]:-lim+MinDirCum[0,j],lim+MinDirCum[1,j]:-lim+MinDirCum[1,j]]
    
    lims = [[int(max(min(DendArr[:,0]-lim2),0)),int(min(max(DendArr[:,0]+lim2),490))],[int(max(min(DendArr[:,1]-lim2),0)),int(min(max(DendArr[:,1]+lim2),490))]]
    t0new,t1new,cut0,cut1 = ExtraStackDel(t0, t1,lims)

    Tnew = [np.delete(T[0],cut0,axis=0) ,np.delete(T[1],cut1,axis=0)]

    for i in range(2,nSnaps):
        t0 =T[i][:,lim+MinDirCum[0,i]:-lim+MinDirCum[0,i],lim+MinDirCum[1,i]:-lim+MinDirCum[1,i]]
        t1new,_,cut0,cut1 = ExtraStackDel(t0,t1new,lims)
        if(cut1 != []):
            for j,t in enumerate(Tnew):
                Tnew[j] = np.delete(Tnew[j],cut1,axis=0) 
        Tnew.append(np.delete(T[i],cut0,axis=0))
    
    Tnew = np.asarray(Tnew)

    return np.moveaxis(Tnew,[0],[-1])

def ExtraStackDel(t0,t1,lims):
    
    """
    Input:
            t0 (np.array of doubles)  : Pixel values at one time step
            t1 (np.array of doubles)  : Pixel values at next time step

    Output:
            t0new  (np.array of doubles) : Trimmed pixel values at one time step
            t1new  (np.array of doubles) : Trimmed pixel values at next time step
            cut0  (list of integers) : Location of trimmings for t0
            cut1  (list of integers) : Location of trimmings for t1

    Function:
            Works out frobenius norm for each stack pairing to work out start
            and end so these can be trimmed
    """
    
    tmat = []
    lims = np.asarray(lims)
    for t0d in t0:
        tdif = []
        for td in t1:
            tdif.append(np.sum((t0d-td)**2)+np.sum((t0d-td)[lims[1,0]:lims[1,1]+1,lims[0,0]:lims[0,1]+1]**2))
        tmat.append(tdif)
    tmat = np.asarray(tmat)

    a = np.zeros(len(t0),dtype=int)
    l = []
    for i,tvec in enumerate(tmat):
        l.append(np.min(tvec))
        tmin = min(i for i in tvec if i > 0)
        a[i] = np.where(tvec==tmin)[0]
    dminV = np.where(a==0)[0]
    dminV = dminV[2*dminV<max(t1.shape[0],t0.shape[0])]
    
    cut1 = []
    if(dminV.size==0):
        dmin = 0
        cut1 = list(range(0,a[dmin]))
        cut0 = []
        l0 = len(t0)
        l1 = len(t1[a[dmin]:])
        if(l0>l1):
            cut0=cut0+list(range(l1,l0))
            t0new = t0[:l1]
            t1new = t1[a[dmin]:]
        elif(l1>l0):
            cut1=cut1+list(range(l0,l1))
            t0new = np.copy(t0)
            t1new = t1[a[dmin]:(l0-l1)]
        else:
            t0new = np.copy(t0)
            t1new = t1[a[dmin]:]
    else:
        dmin = np.where(a==0)[0][np.argmin([l[i] for i in dminV])]
        cut0 = list(range(0,dmin))
        cut1 = []
        l0 = len(t0[dmin:])
        l1 = len(t1)
        if(l0>l1):
            cut0=cut0+list(range(l1,l0))
            t0new = t0[dmin:(l1-l0)]
            t1new = np.copy(t1)
        elif(l1>l0):
            cut1=cut1+list(range(l0,l1))
            t0new = t0[dmin:]
            t1new = t1[:l0]
        else:
            t0new = t0[dmin:]
            t1new = np.copy(t1)
    return t0new,t1new,cut0,cut1

class plotStack:
    def __init__(self,tiff_Arr,SynArr,MinDirCum,SingleClick,Snapshot,Dir):
        self.tiff_Arr = tiff_Arr
        self.SynArr = SynArr
        self.MinDirCum = MinDirCum
        self.SingleClick = SingleClick
        self.Snapshot = Snapshot
        self.GenGif(tiff_Arr,SynArr,Dir)
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.contourf(self.tiff_Arr[0,:,:])
        axfreq = plt.axes([0.25, 0.025, 0.6, 0.03])
        self.sdist = Slider(axfreq, 'Stack', 1, tiff_Arr.shape[0], valinit=1, valstep=1)
        self.sdist.on_changed(self.ChangeStack)
        self.PlotSyns(1)
        plt.show()

        return None
    
    def GenGif(self,tiff_Arr,SynArr,Dir):
        imageNames = []
        for i,t in enumerate(tiff_Arr):
            self.fig, self.ax = plt.subplots(figsize=(10,5))
            self.ax.contourf(t)
            self.PlotSyns(i+1)
            plt.savefig(Dir+'temp_'+str(i)+".png")
            imageNames.append(Dir+'temp_'+str(i)+".png")
            plt.close()
        clip = ImageSequenceClip(imageNames, fps=len(tiff_Arr)//10)
        clip.write_gif(Dir+'SpineSelection_'+str(self.Snapshot)+'.gif', fps=len(tiff_Arr)//10)
        
        for name in imageNames:
            if os.path.exists(name):
                os.remove(name)
        
        return 0
        
    def PlotSyns(self,stack):
        i=0
        for Synapse in self.SynArr:
            if(Synapse.stack==(stack-1)):
                if(self.SingleClick):
                    if(Synapse.xpert.ndim==2):
                        xplt = np.vstack([Synapse.xpert,Synapse.xpert[0]])+self.MinDirCum[::-1,self.Snapshot]
                    else:
                        xplt = np.vstack([Synapse.xpert[0],Synapse.xpert[0][0]])+self.MinDirCum[::-1,self.Snapshot]
                    self.ax.plot(xplt[:,0],xplt[:,1],'k')
                else:
                    t = np.linspace(0,2*np.pi,100)
                    if(self.Snapshot>0):
                        p1 = Synapse.location+self.MinDirCum[::-1,self.Snapshot]
                    else:
                        p1 = Synapse.location
    
                    t = np.linspace(0,2*np.pi,100)
                    x0 = p1[0]+ Synapse.width*np.cos(t)
                    y0 = p1[1]+ Synapse.height*np.sin(t)
                    if(i==0):
                        self.ax.plot(x0,y0,'k')
                        self.ax.annotate(str(i),(p1[0]+5,p1[1]+5),color='black')
                    else:
                        self.ax.plot(x0,y0,'r')
                        self.ax.annotate(str(i),(p1[0]+5,p1[1]+5),color='red')
                    i+=1
        return 0
    def ChangeStack(self,val):

        stack = int(self.sdist.val)
        self.ax.clear()
        self.ax.contourf(self.tiff_Arr[stack-1,:,:])
        self.PlotSyns(stack)
        self.fig.canvas.draw_idle()

        return None