
import sys
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class Simulation:

    """Class that holds the parameters associated with the simulation"""
    def __init__(self,Unit,bgmean,Dir,Snapshots,Mode,z_type,frame=None):
        self.Pic_Type    = 'MAX'
        self.Unit = 0.06589
        self.bgmean    = []
        self.Dir       = Dir
        self.SomaSim   = False
        self.Snapshots = Snapshots
        self.MinDirCum = []

        self.model     = "1stModel"

        self.Times    = []

        self.SingleClick   = True
        self.frame     = frame
        
        if z_type == "Sum":
            self.z_type = np.sum
        elif z_type == "Max":
            self.z_type = np.max
        elif z_type == "Min":
            self.z_type = np.min
        elif z_type == "Mean":
            self.z_type = np.mean
        elif z_type == "Median":
            self.z_type = np.median
        elif z_type == "Std":
            self.z_type = np.std
        else:
            self.z_type = None

        self.Mode    = Mode

class Clicker:
    """Abstract base class for Clicker functions"""
    def __init__(self,frame=None):
        self.key = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.OnClick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.frame    = True if frame == None else False


    def Cancel(self,event):

        """ Function for plot buttons to cancel the program"""

        sys.exit()

    def Close(self,event):

        """ Function for plot buttons to close the plot """
        plt.close()

    def on_key_press(self,event):
        if(event.key=='backspace'):
            self.key = 'backspace'
        elif(event.key=='shift'):
            self.key = 'shift'
        else:
            self.key = None

    def on_key_release(self, event):
        self.key = None

    def OnClick(self,event):
        raise NotImplementedError("Needs to be implemented by a subclass")

    def RoiPlot(self):
        raise NotImplementedError("Needs to be implemented by a subclass")

def getAngle(a,b,c):

    """
    Input:
           a,b,c ( 2 x doubles) : points of interest
    Output:
            angle (double)      : angle at b

    Function:
            Find angle based on cosine rule to find angle
    """

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))

def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://www.geomalgorithms.com/algorithms.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)

def angle_between(v1, v2):

    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between2(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def unitVector(a):
    return a/np.linalg.norm(a)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def projection(p1, p2, p3):

    """ Function that returns the projection of a point onto a line"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy

    a = (dy*(y3-y1)+dx*(x3-x1))/det

    return x1+a*dx, y1+a*dy

def GetPerpendicularVector(a,b):
    u = a-b
    u_p = u/np.linalg.norm(u)

    d_p = np.array([[0, 1], [-1, 0]])
    return d_p.dot(u_p)
    
def crosslen(a0, a1, b0,b1):

    """ Function that returns the projection of a point onto a line"""
    a00, a01 = a0
    a10, a11 = a1
    b00, b01 = b0
    b10, b11 = b1

    lam = ((a01-b01)*(b10-b00)-(a00-b00)*(b11-b01))/((a10-a00)*(b11-b01)-(a11-a01)*(b10-b00))
    mu  = ((a00-b00)+lam*(a10-a00))/(b10-b00)

    if(np.isnan(mu) or np.isinf(mu)):
        mu= ((a01-b01)+lam*(a11-a01))/(b11-b01)

    return lam,mu