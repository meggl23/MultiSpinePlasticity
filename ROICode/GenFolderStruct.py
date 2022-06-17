import os
import shutil
import tifffile as tf
import regex as re


def CreateCellDirs(SourceDir,TargetDir,Name,frame=None):
    """Function that generates the folders with the correct files in the chosen target directory"""

    TargetDir = TargetDir+'/'
    SourceDir = SourceDir+'/'
    
    Name = Name+'/'
    os.mkdir(TargetDir+Name)
    
    i=1
    regex = re.compile('.\d+')
    f = open(TargetDir+Name+"Index.txt", "a")
    for fold in sorted(os.listdir(SourceDir)):
        if(os.path.isdir(SourceDir+fold)):
            os.mkdir(TargetDir+Name+'cell_'+str(i))
            f.write('cell_'+str(i)+':'+fold + '\n')
            for x in os.listdir(SourceDir+fold):
                if(('lsm' in x or 'tif' in x) and re.findall(regex,x)):
                    path1 = SourceDir+fold+'/'+x
                    path2 = TargetDir+Name+'cell_'+str(i)+'/'+x
                    shutil.copyfile(path1, path2)
            i+=1
        if((not frame==None )and i%2==0):
                frame.progress['value'] = (i/len(os.listdir(SourceDir)))*100
                frame.master.update() 
    
    f.close()

def SplitStack(Dir, Files = ['-15','-10','-5','+2','+10','+20','+30','+40']):
    
    regex = re.compile('.\d+')
    for x in os.listdir(Dir):
        try:
            for y in os.listdir(Dir+'/'+x):
                
                if(' to ' in y):
                    l = re.findall(regex,y)
                    start = Files.index(l[0])
                    T = tf.imread(Dir+'/'+x+'/'+y).squeeze()
                    for i,t in enumerate(T):
                        tf.imsave(Dir+'/'+x+'/'+Files[start+i]+'.tif',t)
                    os.remove(Dir+'/'+x+'/'+y)
        except:
            pass
            
        
if __name__ == '__main__':
    #Dir = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/_RAW_DATA_collaboration/2021_7_20 Data/CA1 1x sham LTP'
    #Dir2 = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/NewCode/CA1_1spine'
    #Name='Sham'
    #CreateCellDirs(Dir,Dir2,Name)
    
    Dir = '/Users/maximilianeggl/Dropbox/PostDoc/HeteroPlast/heterosynaptic_project/NewCode/CA1_15spine/Sham'
    SplitStack(Dir)