import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox,Button
from matplotlib.animation import FuncAnimation, PillowWriter

from matplotlib import rc
import ast
plt.rc('text', usetex=False)
font = {'family' : 'serif',
        'size'   : 14}
plt.rc('font', **font)


def RHS(C,P,dx,CParams,PParams,FParams):
    
    return CRHS(C,dx,CParams),PRHS(P,C,dx,PParams),SRHS(P,C,FParams)

def CRHS(C,dx,CParams):
    
    a1 = CParams[0]
    a2 = CParams[1]
    
    return a1*diff2(C,dx) - a2*C

def PRHS(P,C,dx,PParams):
    
    b1 = PParams[0]
    b2 = PParams[1]
    b3 = PParams[2]
    
    return b1*C + b2*diff2(P,dx) - b3*P

def SRHS(P,C,FParams):
    
    NParams = FParams[:2]
    FP = F(P,NParams)

    z1 = FParams[2]
    z2 = FParams[3]

    return z1*C + z2*FP

def diff2(y,dx):
    
    d2C = np.zeros_like(y)
    
    d2C[1:-1] = (np.roll(y,1)[1:-1]+np.roll(y,-1)[1:-1]-2*y[1:-1])/(dx*dx)
    
    return d2C


def F(P,FParams):
    
    nu1 = FParams[0]
    nu2 = FParams[1]
    return P*(np.exp(-(P-nu1)**2) - np.exp(-(P-nu2)**2))

def Simulate(Params,StimPts = np.array([]),StimTime = np.array([])):

    dx = 0.01

    a1 = Params[0]
    a2 = Params[1]

    b1 = Params[2]
    b2 = Params[3]
    b3 = Params[4]

    n1 = Params[5]
    n2 = Params[6]

    z1 = Params[7]
    z2 = Params[8]

    T = int(Params[9])

    dt = dx*dx/(4*max(a1,b2))
    
    x = np.arange(-2,2,dx)
    
    C = np.zeros_like(x)
    P = np.zeros_like(C)
    S = np.zeros_like(C)
    
    C_tot = []
    P_tot = []
    S_tot = []

    CParams = [a1,a2]
    PParams = [b1,b2,b3]
    FParams = [n1,n2,z1,z2]
    i=0
    t = 0
    while t<T:
        R = RHS(C,P,dx,CParams,PParams,FParams)
        C = C + dt*R[0]
        P = P + dt*R[1]
        S = S + dt*R[2]
        if t in StimTime:
            j = np.where(t==StimTime)[0]
            for jj in j:
                C += np.exp(-100*(x-StimPts[jj])**2)
        C_tot.append(C)
        P_tot.append(P)
        S_tot.append(S)
        i+=1
        t = i*dt
    C_tot = np.array(C_tot)
    P_tot = np.array(P_tot)
    S_tot = np.array(S_tot)


    return C_tot,P_tot,S_tot


#===============================================================================

class SystemVideo():
    
    def __init__(self):

        self.fig, axs = plt.subplots(2,2,figsize=(12,8))
        self.ax = axs.ravel()
        self.ax[0].set_title('Calcium',fontdict={'fontsize':14})
        self.ax[1].set_title('P',fontdict={'fontsize':14})
        self.ax[2].set_title('F(P)',fontdict={'fontsize':14})
        self.ax[3].set_title('S',fontdict={'fontsize':14})

        for a in self.ax:
            a.set_ylim([-1,1])
        plt.subplots_adjust(right=0.6)
        self.t = np.arange(-2.0, 2.0, 0.01)
        s = np.zeros_like(self.t)
        self.line, = self.ax[0].plot(self.t, s, lw=2,c='b')
        self.line2, = self.ax[1].plot(self.t, s, lw=2,c='g')
        self.line3, = self.ax[2].plot(self.t, s, lw=2,c='r')
        self.line4, = self.ax[3].plot(self.t, s, lw=2,c='k')

        self.StimPts = []
        self.StimTimes = []

        #                  a1 a2 b1 b2 b3 n1 n2 z1 z2 T
        self.ButtonDict = [0.0001, 0.001, 0.1, 0.001, 0.05, 1, 0.5, 0.1, 0.1,40]
        x = 0.65
        y = 0.9
        plt.figtext(x,y, r'$\frac{\partial C}{\partial t} = \qquad \qquad \frac{\partial^2 C}{\partial x^2} - \qquad \qquad C $',size=18)

        axbox = plt.axes([x+0.044,y - 0.016 , 0.075, 0.05])
        a1Box = TextBox(axbox, '', initial='a1', textalignment="center")
        a1Box.on_submit(lambda val: self.submit(val, 0))
        axbox = plt.axes([x + 0.18, y - 0.016, 0.075, 0.05])
        a2Box = TextBox(axbox, '', initial='a2' , textalignment="center")
        a2Box.on_submit(lambda val: self.submit(val, 1))


        y = y-0.1

        plt.figtext(x, y, r'$\frac{\partial P}{\partial t} = \qquad \qquad C + \qquad \qquad \frac{\partial^2 P}{\partial x^2}$',size=18)
        plt.figtext(x+ 0.04, y-0.07, r'$- \qquad \qquad P$',size=18)
        axbox = plt.axes([x+0.044,y - 0.016 , 0.075, 0.05])
        b1Box = TextBox(axbox, '', initial='b1' , textalignment="center")
        b1Box.on_submit(lambda val: self.submit(val, 2))

        axbox = plt.axes([x + 0.17, y - 0.016, 0.075, 0.05])
        b2Box = TextBox(axbox, '', initial='b2' , textalignment="center")
        b2Box.on_submit(lambda val: self.submit(val, 3))

        axbox = plt.axes([x +0.06 , y - 0.086, 0.075, 0.05])
        b3Box = TextBox(axbox, '', initial='b3' , textalignment="center")
        b3Box.on_submit(lambda val: self.submit(val, 4))

        y = y-0.15

        plt.figtext(x,y, r'$F(P) = P (e^{-(P-  \qquad \qquad )^2} $',size=16)
        plt.figtext(x+ 0.06, y-0.07, r'$- e^{-(P-\qquad \qquad )^2} ) $',size=16)
        
        axbox = plt.axes([x+0.13, y+0.005, 0.05, 0.03])
        n1Box = TextBox(axbox, '', initial='n1' , textalignment="center")
        n1Box.on_submit(lambda val: self.submit(val, 5))

        axbox = plt.axes([x+0.125, y-0.065, 0.05, 0.03])
        n2Box = TextBox(axbox, '', initial='n2' , textalignment="center")
        n2Box.on_submit(lambda val: self.submit(val, 6))


        y = y-0.15

        plt.figtext(x,y, r'$\frac{\partial S}{\partial t} = \qquad \qquad C + \qquad \qquad P $',size=18)

        axbox = plt.axes([x+0.044,y - 0.016 , 0.075, 0.05])
        z1Box = TextBox(axbox, '', initial='z1', textalignment="center")
        z1Box.on_submit(lambda val: self.submit(val, 7))
        axbox = plt.axes([x + 0.17, y - 0.016, 0.075, 0.05])
        z2Box = TextBox(axbox, '', initial='z2' , textalignment="center")
        z2Box.on_submit(lambda val: self.submit(val, 8))

        y = y-0.1
        plt.figtext(x,y, r'From $t=0$ to ',size=18)
        axbox = plt.axes([x+0.15,y - 0.016 , 0.075, 0.05])
        TBox = TextBox(axbox, '', initial='T', textalignment="center")
        TBox.on_submit(lambda val: self.submit(val, 9))

        y = y-0.1
        plt.figtext(x,y, r'Stimulations (x,t): ',size=18)
        axbox = plt.axes([x,y - 0.06 , 0.2, 0.05])
        StimBox = TextBox(axbox, '', initial='(,)', textalignment="center")
        StimBox.on_submit(self.Stims)

        nax = plt.axes([0.725, 0.12, 0.1, 0.04])
        kbutton = Button(nax, 'Run!', hovercolor='0.5')
        #print(t1.get_val())
        kbutton.on_clicked(self.Run)
        plt.show()

    def submit(self,val,id):
        try:
            self.anim.event_source.stop()
        except:
            pass
        self.ButtonDict[id] = float(val)

    def Stims(self,val):
        self.StimPts = []
        self.StimTimes = []
        try:
            self.anim.event_source.stop()
        except:
            pass
        if( ' ' in val):
            for v in val.split():
                v1 = ast.literal_eval(v)
                self.StimPts.append(v1[0])
                self.StimTimes.append(v1[1])
        else:
            self.StimPts.append(ast.literal_eval(val)[0])
            self.StimTimes.append(ast.literal_eval(val)[1])
        self.StimPts = np.array(self.StimPts)
        self.StimTimes = np.array(self.StimTimes)
    def Run(self,event):
        try:
            self.anim.event_source.stop()
        except:
            pass
        self.C,self.P,self.S = Simulate(self.ButtonDict,self.StimPts,self.StimTimes)
        self.FP = F(self.P,self.ButtonDict[5:7])
        self.anim = FuncAnimation(self.fig, self.animate, self.C.shape[0]//100,blit=True)
        self.anim.save("movie.gif", writer=PillowWriter(fps=200))
        plt.show()

    
    def animate(self,i):
        self.line.set_ydata(self.C[i*100])
        self.line2.set_ydata(self.P[i*100])
        self.line3.set_ydata(self.FP[i*100])
        self.line4.set_ydata(self.S[i*100])
        return self.line,self.line2,self.line3,self.line4,


if __name__ == '__main__':

    SV = SystemVideo() 