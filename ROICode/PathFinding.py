import numpy as np
import queue
from skimage.draw import line

from Utility import *

def FindSoma(mesh,start,end,verbose=False,scale=0.114):

    """
    Input:
            mesh (np.array)  : Bool array of tiff file > some threshold
            start (ints)     : Start of path
            end (ints)       : end of path
            Verbose (Bool)   : Wether you want the program to tell you whats going on
    Output:
            Path along the mesh from start to end

    Function:
            Uses the Breadth-first algorithm to explore the mesh
    """

    count = 0;
    add = ""
    maze  = mesh
    visited = np.zeros_like(maze)
    direction = np.zeros_like(maze)

    nums = queue.Queue()
    j,i= start;
    if valid(maze, visited,start):
        nums.put(start)
    else:
        print("choose valid starting cell")

    endlist = end.tolist()
    visited[j,i] = 1;
    while not nums.empty(): 
        count += 1;
        add = nums.get()
        if (add.tolist() in endlist or (add==end).all()):
            return GetPath(start,add,direction)

        for j,dirc in np.array([[[0,1],1], [[0,-1],2], [[-1,0],3], [[1,0],4],[[1,1],5], [[1,-1],6], [[-1,1],7], [[-1,-1],8]]):
            put = add + j
            if valid(maze, visited,put):
                count +=1;
                j,i = put;
                visited[j,i] =1
                direction[j,i] = dirc;
                nums.put(put)

    return [start[1],start[0]],0

def valid(maze, visited,moves):
    j,i = moves
    if not(0 <= i < len(maze[0]) and 0 <= j < len(maze)):
        return False
    elif (maze[j][i] == 0):
        return False
    elif ((visited[j][i] == 1)):
        return False
    return True

def GetPath(start,end,directions,scale=0.114,shorten=False):

    """
    Input:
            start (ints)     : Start of path
            end (ints)       : end of path
            mesh (np.array)  : Directions on mesh that point towards the start
    Output:
            path and length of shortest path

    Function:
            Propagates the found directions back from the end to the start
    """

    # 4 for Up, 3 for Down, 2 for right and 1 for left
    current = end
    path_arr = [np.array([end[1],end[0]])]
    j,i= current
    length = 0;
    fp = []
    sp = []
    while not (current == start).all():
        j,i= current
        if directions[j,i] == 4:
            length += 1;
            current = current - [1,0];
        elif directions[j,i] == 3:
            length += 1;
            current = current - [-1,0];
        elif directions[j,i] == 2:
            length += 1;
            current = current - [0,-1];
        elif directions [j,i] == 1:
            length += 1;
            current = current - [0,1]
        elif directions[j,i] == 5:
            length += np.sqrt(2);
            current = current - [1,1];
        elif directions[j,i] == 6:
            length += np.sqrt(2);
            current = current - [1,-1];
        elif directions[j,i] == 7:
            length += np.sqrt(2);
            current = current - [-1,1];
        elif directions [j,i] == 8:
            length += np.sqrt(2);
            current = current - [-1,-1]
        else:
            break;
            print("there is some error")
        path_arr.append([current[1],current[0]])
    fp,sp = SecondOrdersmoothening(np.asarray(path_arr),np.sqrt(2)/scale)
    return path_arr,length*scale

def SmoothenPath(x,y):
    
    #TODO: Describe function
    
    length = len(x)
    modified_list = [[x[0],y[0]]]
    for i in range(1,length-1):
        A = np.array([x[i]-x[i-1],y[i]-y[i-1]])
        B = np.array([x[i]-x[i+1],y[i]-y[i+1]])
        cp = np.cross(A,B)
        if cp != 0:
            modified_list.append([x[i],y[i]])
    modified_list.append([x[-1],y[-1]])
    modified_list = np.asarray(modified_list)
    return modified_list

def SecondOrdersmoothening(orig_path,min_dist):
    
    #TODO: Describe function
    
    first_order = SmoothenPath(orig_path[:,0],orig_path[:,1])
    second_order = [first_order[0]]
    for vdx,v in enumerate(first_order[0:-2]):
        if dist(v,first_order[vdx+1]) > min_dist:
            second_order.append(first_order[vdx+1])
    if not (second_order[-1] == first_order[-1]).all():
        second_order.append(first_order[-1])
    return np.asarray(second_order),first_order

def PathLength(xys,scale):
    
    #TODO: Describe function
    
    length = 0 ;
    for idx in range(xys.shape[0]-1):
        length += dist(xys[idx], xys[idx+1])
    return length*scale

def GetAllpointsonPath(xys):
    
    #TODO: Describe function
    points = np.array([xys[0]])
    for idx in range(xys.shape[0]-1):
        a = xys[idx]
        b = xys[idx+1]
        rr,cc = line(a[0],a[1],b[0],b[1])
        points = np.concatenate((points,(np.column_stack((rr,cc)))))
    new_array = np.array([tuple(row) for row in points])
    _, idx = np.unique(new_array,axis=0,return_index=True)
    return new_array[np.sort(idx)]