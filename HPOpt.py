import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os
import json

import math

class ChemEquations():
    """Class that holds the parameters associated with governing equations"""
    def __init__(self,C0,P0,alpha,b1,b2,c1,c2,nu1,nu2):

        self.C_init = C0
        self.P_init = P0
        self.a  = alpha
        self.b1 = b1
        self.b2 = b2

        self.c1 = c1
        self.c2 = c2

        self.nu1 = nu1
        self.nu2 = nu2

class TS_euler:

    """Class that defines the forward euler step """

    def RHS_exp_fw(self,Sol,CE,dt):

        """
        Input:
                Sol (np.array of doubles)  : values of C, P , S
                CE  (class )               : Governing equation parameters
                dt  (double)               : Time step
        Output:
                np.array of the next C,P and S
        
        Function:
               Using an euler step, solve the governing equations and step 
               C,P and S forward
        """

        # Some of this could be done analytically
        C       = Sol[0]
        P       = Sol[1]
        S       = Sol[2]

        C_RHS   =   - CE.a*C  
        P_RHS   =   + CE.b1*C - CE.b2*P 

        FP      = P*(np.exp(-(P-CE.nu1)**2) - np.exp(-(P-CE.nu2)**2))

        S_RHS   = CE.c1*C + CE.c2*FP

        C_RHS   = C   + dt*C_RHS
        P_RHS   = P   + dt*P_RHS
        S_RHS   = S   + dt*S_RHS

        return np.array([C_RHS,P_RHS,S_RHS])

     def RHS_exp_bw(self,AdjSol,CE,dt,force,Sol):

        """
        Input:
                AdjSol (np.array of doubles)  : values of adjoint C, P , S
                CE  (class )                  : Governing equation parameters
                dt  (double)                  : Time step
                force (double)                : Forcing term coming from cost function
                Sol (np.array of doubles)     : values of C, P , S
        Output:
                np.array of the next adjoint C,P and S
        
        Function:
               Using an euler step, solve the adjoint equations and step 
               adjoint C,P and S forward
        """
        C_adj  = AdjSol[0]
        P_adj  = AdjSol[1]
        S_adj  = AdjSol[2]

        P      = Sol

        dFP    =  (np.exp(-(P-CE.nu1)**2) - np.exp(-(P-CE.nu2)**2)) 
                    - P*((P-CE.nu1)*np.exp(-(P-CE.nu1)**2) 
                        - (P-CE.nu2)*np.exp(-(P-CE.nu2)**2))

        C_RHS  =   + CE.a*C_adj  - CE.b1*P_adj - CE.c1*S_adj
        P_RHS  =   - CE.c2*S_adj*dFP + CE.b2*P_adj
        S_RHS  = force

        C_RHS  = C_adj  + dt*C_RHS
        P_RHS  = P_adj  + dt*P_RHS
        S_RHS  = S_adj  + dt*S_RHS

        return np.array([C_RHS,P_RHS,S_RHS])
   


def ForwardSolve(TS,CE,dt,tpts):

    """
    Input:
            TS (timestepper)     : Needs a forward function
            CE  (class )         : Governing equation parameters
            dt  (double)         : Time step
            tpts (int)           : Number of steps we take
    Output:
            CSave,PSave,SSave    : Arrays that hold full time history
    
    Function:
           Loop through time to get time dynamics of C,P and S
    """

    C   = CE.C_init
    P   =  CE.P_init
    S   = 1
    Sol = np.array([C,P,S])

    CSave = []
    PSave = []
    SSave = []

    for i in range(tpts):

        Sol = TS.RHS_exp_fw(Sol,CE,dt)

        CSave.append(Sol[0])
        PSave.append(Sol[1])
        SSave.append(Sol[2])

    return CSave,PSave,SSave

def BackwardSolve(TS,CE,dt,tpts,force,pSol):

    """
    Input:
            TS (timestepper)     : Needs a backward function
            CE  (class )         : Governing equation parameters
            dt  (double)         : Time step
            tpts (int)           : Number of steps we take
            force (double)       : Forcing term coming from cost function 
            pSol (np.array)      : Solution from forward run to pass to the adjoint
    Output:
            CSave,PSave,SSave    : Arrays that hold full time history of the adjoint
    
    Function:
           Loop through time to get time dynamics of adjoint C,P and S
    """

    dt = -dt
    C  = 0 
    P  = 0 
    S  = 0

    AdjSol = np.array([C,P,S])
    CSave  = []
    PSave  = []
    SSave  = []

    for i in range(tpts):

        AdjSol = TS.RHS_exp_bw(AdjSol,CE,dt,force[i],pSol[i])

        CSave.append(AdjSol[0])
        PSave.append(AdjSol[1])
        SSave.append(AdjSol[2])

    return CSave,PSave,SSave

def Cost(Sol,tSol):

    """
    Input:
            Sol  (np.array) : Solution from Simulation
            tSol (np.array) : Solution to be learned
    Output:
            cost (double)
    
    Function:
           Find L2 norm of difference between simulation and optimal dynamics
    """
    
    return np.sum((Sol-tSol)**2)


def grad(dt,AdjSol,Sol,CE):

    """
    Input:
            dt  (double)       : Time step
            Sol  (np.array)    : Solution from Simulation
            AdjSol  (np.array) : Solution from adjoint Simulation
            CE  (class )       : Governing equation parameters
    Output:
            Gradient (list)    : list of gradients for each parameter
    
    Function:
           As derived from the adjoint framework, we here calculate the 
           gradient updates as necessary to improve our guess
    """

    CAdj  = AdjSol[0]
    PAdj  = AdjSol[1]
    SAdj  = AdjSol[2]

    C     = Sol[0]
    P     = Sol[1]
    S     = Sol[2]

    FP    = P*(np.exp(-(P-CE.nu1)**2) 
            - np.exp(-(P-CE.nu2)**2))

    dC0   = CAdj[-1]

    dP0   = PAdj[-1]

    dFdnu1 =  2 * P*np.exp(-(P-CE.nu1)**2)*(P-CE.nu1)

    dFdnu2 = -2 * P*np.exp(-(P-CE.nu2)**2)*(P-CE.nu2)

    da  =  -dt * np.sum(C*CAdj)

    db1 =   dt * np.sum(C*PAdj)
    db2 =  -dt * np.sum(P*PAdj)

    #dc1 =  dt * np.sum(C*SAdj) - here for completness, but it is actually redundant
    dc2 =  dt * np.sum(FP*SAdj)

    dnu1 =  dt * CE.c2 * np.sum(SAdj*dFdnu1)
    dnu2 =  dt * CE.c2 * np.sum(SAdj*dFdnu2)

    grad = [dC0,dP0,da,db1,db2,0,dc2,dnu1,dnu2]


    return grad

def TrueSolution(tpts,CEtrue,dt):

    """
    Input:
            dt  (double)       : Time step
            CEtrue  (class )   : Governing equation parameters of true values
            tpts (int)         : Number of steps we take

    Output:
            Gradient (np.array): The output given the true parameters
    
    Function:
           Calculate the dynamics given true equation parameters
    """


    Ts = TS_euler()

    SolTrue = ForwardSolve(Ts,CEtrue,dt,tpts)

    return SolTrue

class AdjointSolver:

    """Class that defines the optimisation loop """ 

    def __init__(self,tpts,dt,tSol,tmax,optchoice,TrueDat=False):
        self.tpts = tpts
        self.dt   = dt
        self.tSol  = tSol
        self.tmax  = tmax
        self.TrueDat = TrueDat

        self.OptChoice = optchoice


    def singleloop(self,params):

        """
        Input:
                params (np.array or list) : list of parameters for the simulation 
                                            for that given list

        Output:
                cost     (double)  : Cost of current parameters minus true solution
                update (np.array)  : How to update parameters to do better
        
        Function:
               Full loop to be incorporated in optimisation routine
        """
        
        tSol = self.tSol
        dt   = self.dt
        tmax = self.tmax

        C0     = params[0]
        P0     = params[1]
        
        alpha  = params[2]
        
        b1     = params[3]
        b2     = params[4]

        c1     = params[5]
        c2     = params[6]

        nu1    = params[7]
        nu2    = params[8]
        

        CE = ChemEquations(C0,P0,alpha,b1,b2,c1,c2,nu1,nu2)

        Ts = TS_euler()

        cSol,pSol,sSol = ForwardSolve(Ts,CE,self.dt,self.tpts)


        cSol.reverse()
        pSol.reverse()
        sSol.reverse()

        if(self.TrueDat):
            cost = 0 
            tvec  =  np.round(np.linspace(self.tmax,0,self.tpts),3)
            tsnaps = []
            force = np.zeros_like(sSol)
            for d,t in zip(self.tSol[0],self.tSol[1]):
                force[t==tvec] = 2*(np.nanmean(np.asarray(sSol)[t==tvec]-d))
                tsnaps.append(np.argwhere(t==tvec)[0][0])
                cost   += np.nanmean(((np.asarray(sSol)[t==tvec]-d)**2))
        else:
            force = 2*(np.asarray(sSol)-np.asarray(tSol))
            cost    = dt*Cost(sSol,tSol)  

        TS = TS_euler()

        acSol,apSol,asSol = BackwardSolve(TS,CE,self.dt,self.tpts,force,np.asarray(pSol))

        cSol   = np.asarray(cSol)
        pSol   = np.asarray(pSol)
        sSol   = np.asarray(sSol)

        acSol   = np.asarray(acSol)
        apSol   = np.asarray(apSol)
        asSol   = np.asarray(asSol)

        gradvec = grad(dt,[acSol,apSol,asSol],[cSol,pSol,sSol],CE)
            
        update  = np.array(gradvec)
        return cost,-update

def DatRun(Dat,tSnaps,Dir,InitMethod,optchoice):

    """
        Input:
                Dat    (np.array)    :  Output from experiments to learn
                tSnaps (list )       :  Snapshot times
                Dir    (string)      :  Directory where to look for seed files or save pictures
                InitMethod (string)  :  Wether we start from a file or randomly
                optchoice (list)     :  Which parameters we want to optimize

        Output:
                N/A
        
        Function:
               Utilizing the single loop, uses the scipy.optimize package to learn the parameters
    """

    tmin    =  0
    tmax    =  40
    tpts    =  tmax*1000+1
    dt     = (tmax-tmin)/tpts
    tvec  = np.linspace(tmin,tmax,tpts)

    AS = AdjointSolver(tpts,dt,[Dat,tSnaps],tmax,optchoice,True)
    
    if(InitMethod=='File'):
        CE = InitSystem(Dir,50,Dat,tSnaps,dt,tpts,tmax)
    else:
        CE = Primer(50,Dat,tSnaps,dt,tpts,tmax)
    
    C0    = CE.C_init
    P0    = CE.P_init

    alpha = CE.a

    b1    = CE.b1
    b2    = CE.b2

    c1    = CE.c1
    c2    = CE.c2

    nu1   = CE.nu1
    nu2   = CE.nu2

    PVec = np.array([C0,P0,alpha,b1,b2,c1,c2,nu1,nu2])

    if(not SeedFile == None ):
        boundvec = np.array([(0,np.inf)]*9)
        boundvec[~optchoice,0] = PVec[~optchoice]*0.9
        boundvec[~optchoice,1] = PVec[~optchoice]*1.1
    else:
        boundvec = np.array([(0,np.inf)]*9)
        
    res = optimize.minimize(AS.singleloop, np.array([C0,P0,alpha,b1,b2,c1,c2,nu1,nu2]),bounds=boundvec, options={'ftol':1e-16,'gtol': 1e-16,'disp':True,'maxls':20},jac=True,method='L-BFGS-B',tol=1e-16)

    C0,P0,alpha,b1,b2,c1,c2,nu1,nu2    = res.x

    cost = res.fun 

    TS = TS_euler()

    CEest = ChemEquations(C0,P0,alpha,b1,b2,c1,c2,nu1,nu2)

    cSol,pSol,sSol = ForwardSolve(TS,CEest,dt,tpts)

    plt.plot(tvec,sSol)
    plt.boxplot(Dat.T,positions=tSnaps,showfliers=False)
    plt.ion()
    plt.show()
    SaveSystem(CEest,Dir,cost)

def ReadFiles(Dir):

    """
        Input:
                Dir    (string)      :  Directory where to find expt files

        Output:
                Areas_arr (np.array) : Values of the spine luminosities
                Dist_arr  (np.array) : Distance metrics of the spines
        
        Function:
               Read the json files provided by the SynapseTool
    """

    times = [-15,-10,-5,2,10,20,30,40]
    Accept = []

    for d in os.listdir(Dir):
        try:
            if('Synapse_l.json' in os.listdir(Dir+d) ):
                Accept.append(d)
        except:
            pass

    Syn_a_arr = []
    bg_l      = []

    for a in Accept:
        with open(Dir+a+'/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        try:
            bg_l.append(np.load(Dir+a+"/backgroundM.npy").squeeze())
        except:
            bg_l.append(np.load(Dir+a+"/background.npy").squeeze())

    Areas_arr = []
    Dist_arr  = []
    bg_arr = []

    for Syn_a,bg in zip(Syn_a_arr,bg_l):

        areas = np.array([S["RawIntDen"] for S in Syn_a])
        bg_arr = np.zeros_like(areas)

        for i, Syn in enumerate(Syn_a):
            bg_arr[i,:] = Syn["area"]*bg/(0.066**2)

        if(not Syn_a[0]["Times"]==times):
            for l in list(set(times)-set(Syn_a[0]["Times"]))[::-1]:
                areas = np.insert(areas,times.index(l),math.nan,-1)
                bg_arr = np.insert(bg_arr,times.index(l),math.nan,-1)
                
        dist_a = [S["distance"] for S in Syn_a]
        Dist_arr  = Dist_arr+dist_a
        Areas_arr.append(areas-bg_arr)
        
    Areas_arr = np.vstack(Areas_arr)
    Dist_arr = np.array(Dist_arr)
    Areas_arr = (Areas_arr.T/Areas_arr[:,:3].mean(axis=1)).T

    return Areas_arr,Dist_arr

def Primer(nPrimes,Dat,tSnaps,dt,tpts,tmax):
    
    """
        Input:
               nPrimes  (int) : Number of random initializations
               Dat (np.array) : data from experiment
               tSnaps (list)  : time of snapshots
               dt  (double)   : Time step
               tpts (int)     : Number of steps we take 
               tmax (double)  : Maximum time of expt

        Output:
               CEOld  (class )   : Best initial parameter set

        Function:
               Randomly generate parameters and compare them to the true solution
               The one with the lowest cost gets picked
    """

    
    OldCost = np.inf
    for i in range(nPrimes):
        C0    = np.random.rand()
        P0    = np.random.rand()
        alpha = np.random.rand()

        b1    = np.random.rand()
        b2    = np.random.rand()

        c1    = 1
        c2    = np.random.rand()

        nu1   = np.random.rand()
        nu2   = np.random.rand()

        CE = ChemEquations(C0,P0,alpha,b1,b2,c1,c2,nu1,nu2)

        TS = TS_euler()

        _,_,sSol = ForwardSolve(TS,CE,dt,tpts)

        sSol.reverse()

        cost = 0 
        tvec  =  np.round(np.linspace(tmax,0,tpts),3)
        for d,t in zip(Dat,tSnaps):
            cost   += np.nanmean(((np.asarray(sSol)[t==tvec]-d)**2)) 

        print('Cost of sys   ',i,'   the cost is   ', cost)
        if(cost<OldCost):
            CEOld = CE
            OldCost = cost

    return CEOld


def SaveSystem(CE,Dir,cost):

    """
        Input:
                EOld  (class)   : Governing equation parameters
                Dir (string)    : Location to save system
                cost (double)   : cost of the system

        Output:
                N/A

        Function:
                Given a parameter set, save it to a json file. 
    """

    Cd = CE.__dict__
    Cd['score'] = cost

    try: 
        with open(Dir+'Best_System.json', 'r') as fp: temp = json.load(fp)
        if(temp['score']<Cd['score']):
            print('The original system was better, I print the new system here but dont save it')
            print(Cd)
        else:
            print('The new system is better, I print the old system here but dont save it')
            print(temp)
            with open(Dir+'Best_System.json', 'w') as fp: json.dump(Cd,fp,indent=4)
    except:
        with open(Dir+'Best_System.json', 'w') as fp: json.dump(Cd,fp,indent=4)

    return 0

def InitSystem(Dir,nPrimes,Dat,tSnaps,dt,tpts,tmax):

    """
        Input:
               Dir (string)   :  Directory where to look for seed files
               nPrimes  (int) : Number of random initializations
               Dat (np.array) : data from experiment
               tSnaps (list)  : time of snapshots
               dt  (double)   : Time step
               tpts (int)     : Number of steps we take 
               tmax (double)  : Maximum time of expt

        Output:
               CEOld  (class )   : Return initial governing equation parameters

        Function:
               Decide wether to use a seedfile or the primer
    """

    if(not SeedFile == None ):
        with open(SeedFile, 'r') as fp: temp = json.load(fp)
        return ChemEquations(temp['C_init'],temp['P_init'],temp['a'],temp['b1'],temp['b2'],temp['c1'],temp['c2'],temp['nu1'],temp['nu2'])
    else:
        try:
            with open(Dir+Name+'.json', 'r') as fp: temp = json.load(fp)
            return ChemEquations(temp['C_init'],temp['P_init'],temp['a'],temp['b1'],temp['b2'],temp['c1'],temp['c2'],temp['nu1'],temp['nu2'])
        except:
            t1 = os.listdir(Dir)
            try:
                for x in ['1','3','7','15']:
                    if(x+'_npot.json' in t1):
                        fileName = x+'_pot.json'
                        break
                with open(Dir+fileName, 'r') as fp: temp = json.load(fp)
                return ChemEquations(temp['C_init'],temp['P_init'],temp['a'],temp['b1'],temp['b2'],temp['c1'],temp['c2'],temp['nu1'],temp['nu2'])
            except:
                return Primer(nPrimes,Dat,tSnaps,dt,tpts,tmax)

def PreSortData(RealDat,Flag):

    """
        Input:
               RealDat : Dataset from experiments
               Flag    : Check wether we want potentiation or not

        Output:
                Sorted dataset

        Function:
               If the spine does not move significantly, we class it a failure
    """
    
    Pot = []
    for d in RealDat:
        if(abs((d[3]-d[:3].mean())/d[:3].std())>1.96 and d[3]-d[:3].mean()>0):
            Pot.append(Flag)
        else:
            Pot.append(not Flag)

    return np.delete(RealDat,Pot,axis=0)



def DrugExpt(DirName,Type='beta'):

    """ Function to evaluate drug expt"""

    times = [2,10,20,30,40]
    for N in ['1','7']:
        Name = N+'_'+DirName
        global SeedFile
        try:
            if(N=='1'):
                SeedFile = '1Spine/AIP/1_CamKII_alt_nu.json'
            else:
                SeedFile = '7Spine/AIP/7_CamKII_nu.json'
            
            Dir = N+'Spine/'+DirName+'/'
            A,D = ReadFiles(Dir)
            RealDat = A[D==0]
            InitMethod = 'File'
    
            if(Type=='nu'):
                optchoice = np.array([0,1,0,0,0,0,0,1,1]).astype(bool)
            elif(Type=='beta'):
                optchoice = np.array([0,1,0,1,1,0,0,0,0]).astype(bool)
            elif(Type=='P0'):
                optchoice = np.array([0,1,0,0,0,0,0,0,0]).astype(bool)
            elif(Type=='nus'):
                optchoice = np.array([0,0,0,0,0,0,0,1,1]).astype(bool)
            else:
                optchoice = np.array([0,1,0,0,0,0,0,0,0]).astype(bool)
        
            DatRun(RealDat.T[3:,:],times,Dir,InitMethod,optchoice)    

            with open(Dir+'Best_System.json', 'r') as fp: temp = json.load(fp)
            os.remove(Dir+'Best_System.json')
            with open(Dir+Name+'_'+Type+'.json', 'w') as fp: json.dump(temp,fp,indent=4)

def ClustExpt():

    """ Function to evaluate cluster expt"""

    for N in ['7']:
        print('---------------------------')
        print(N)
        print('---------------------------')
        global SeedFile
        try:
            SeedFile = N+'Spine/Control/'+N+'_pot.json'
            Name = N+'_clust'
            Dir = N+'Spine/Control/'
            A,D = ReadFiles(Dir)
            RealDat = A[D<0]
            times = [2,10,20,30,40]


            InitMethod = 'File'

            optchoice = np.array([1,1,0,0,0,0,0,0,0]).astype(bool)
     
            DatRun(RealDat.T[3:,:],times,Dir,InitMethod,optchoice)    

            with open(Dir+'Best_System.json', 'r') as fp: temp = json.load(fp)
            os.remove(Dir+'Best_System.json')
            with open(Dir+Name+'.json', 'w') as fp: json.dump(temp,fp,indent=4)
        except:
            print('No seed file!')

def DrugExptClust(DirName,Type='beta'):

    """ Function to evaluate drug cluster expt"""

    times = [2,10,20,30,40]
    for N in ['7']:
        Name = N+'_'+DirName
        global SeedFile
        Dir = '7Spine/AIP/'
        try:
            SeedFile = '7Spine/AIP/7_CamKII_nu.json'
            A,D = ReadFiles(Dir)
            RealDat = A[D<0]
            InitMethod = 'File'
            optchoice = np.array([1,1,0,0,0,0,0,0,0]).astype(bool)
        
            DatRun(RealDat.T[3:,:],times,Dir,InitMethod,optchoice)    

            with open(Dir+'Best_System.json', 'r') as fp: temp = json.load(fp)
            os.remove(Dir+'Best_System.json')
            with open(Dir+Name+'_'+Type+'_clust.json', 'w') as fp: json.dump(temp,fp,indent=4)
        except:
            print('No seed file!')

        
def ControlExpt():


    """ Function to evaluate stimulated spines expt"""

    times = [2,10,20,30,40]
    global SeedFile
    SeedFile = None
    optchoice = np.array([0,1,0,0,0,0,0,1,1]).astype(bool)
    for N in ['15']:
        print('---------------------------')
        print(N)
        print('---------------------------')
        global Name
        try:
            SeedFile = N+'spine/Control/'+N+'_pot.json'
            
            Name = N+'_pot'
            Dir = N+'spine/Control/'
            A,D = ReadFiles(Dir)
            RealDat = A[D==0]
            Flag = False
            RealDat = PreSortData(RealDat,Flag)
            InitMethod = 'File'


            DatRun(RealDat.T[3:,:],times,Dir,InitMethod,optchoice)    

            with open(Dir+'Best_System.json', 'r') as fp: temp = json.load(fp)
            os.remove(Dir+'Best_System.json')
            with open(Dir+Name+'.json', 'w') as fp: json.dump(temp,fp,indent=4)
        except:
            print('No seed file!')

    

if __name__ == '__main__':

    Mode = 0 #1,2,3
    if Mode==0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Control')
        ControlExpt()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    elif Mode==1:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Drugs')
        print('---------------------------')
        print('AIP')
        print('---------------------------')
        DrugExpt('AIP',Type='nu')
        print('---------------------------')
        print('FK506')
        print('---------------------------')
        DrugExpt('FK506',Type='nu')
    elif Mode==2:
        print('Cluster')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        ClustExpt()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    elif Mode==3:
        print('Drug Clusters')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        DrugExptClust('AIP',Type='nus')
        DrugExptClust('FK506',Type='nus')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        