#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:22:20 2021

@author: surbhitwagle
"""

import numpy as np
import os
from math import sqrt
import tifffile as tf
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
#from skimage.color import rgb2gray
from os.path import exists
# lsm_img = np.load("lsm_img.npy", mmap_mode='r')
# print("image shape = ",np.shape(lsm_img))         
from skimage.draw import line,polygon,ellipse,disk
from skimage import morphology

class PunctaDetection():
    """
        class that holds meta data for puncta detection and methods for puncta stats calculations
    """
    def __init__(self,SimVars,tiff_Arr,somas,dendrites,channels,width=5):
        self.Dir      = SimVars.Dir
        self.tiff_Arr = tiff_Arr
        self.somas = somas                #should be a dict with name of dendrite as key and polygone as value
        self.dendrites = dendrites        #should be a dict with name of dendrite as key and ROIs as values
        self.channels = channels
        self.scale = SimVars.Unit         #micons/pixel
        self.width = width/self.scale
    def isBetween(self,a,b,c):
        """
            function that checks if c lies on perpendicular space between line segment a to b
            input: roi consecutive points a,b and puncta center c 
            output: True/False 
        """
        sides = np.zeros(3)
        sides[0] = (a[0]-b[0])**2 + (a[1]-b[1])**2  #ab
        original = sides[0]
        sides[1] = (b[0]-c[0])**2 + (b[1]-c[1])**2  #bc
        sides[2] = (c[0]-a[0])**2 + (c[1]-a[1])**2  #ca
        sides = np.sort(sides);
    #     print(sides)
        if sides[2] > (sides[1] + sides[0]) and sides[2] != original:
            return False;
    
        return True;
        
    def Perpendicular_Distance_and_POI(self,a,b,c):
    #     
        """
            distance between two parallel lines, one passing (line1, A1 x + B1 y + C1 = 0) from a and b 
            and second one (line 2, A1 x + B1 y + C2 = 0) parallel to line1 passing from c is given
            |C1-C2|/sqrt(A1^2 + B1^2)
            
            input: roi consecutive points a,b and puncta center c 
            output: Perpendicular from line segment a to b and point of intersection at the segment
        """
        m = (a[1]-b[1])/(a[0]-b[0]+1e-18)
        if m == 0:
            m = 1e-9
        c1 = a[1] - m*a[0]
        c2 = c[1] - m*c[0]
        dist = np.absolute(c1-c2)/np.sqrt(1+m**2)
        m_per = -1/m;
        c3 = c[1] - m_per*c[0]
        x_int = (c3 - c1)/(m-m_per)*1.0
        y_int = (m_per*x_int + c3)*1.0
        
        ax_int = np.sqrt((a[0]-x_int)**2 + (a[1]-y_int))
        bx_int = np.sqrt((b[0]-x_int)**2 + (b[1]-y_int))
        ab = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return x_int,y_int,dist
    
    def GetClosestRoiPoint(self,dendrite,point):
        """
            function that finds closest roi point if point is not on dendrite
            input: dendrite rois,point
            output: distance from the origin of the dendrite
        """
        min_dist = 10**18;
        prev = [dendrite[0][0],dendrite[1][0]]
        dist_from_origin = 0;
        closest_p = [0,0]
        closed_p_idx = 0
    #     print(prev)
        for idx,x in enumerate(dendrite[0][:]):
            y = dendrite[1][idx]
            a = [x,y]
            dist = np.sqrt((point[1]-a[1])**2 + (point[0]-a[0])**2)
            if dist < min_dist:
                min_dist = dist
                dist_from_origin += np.sqrt((prev[1]-a[1])**2 + (prev[0]-a[0])**2)
                closest_p = a
                closed_p_idx = idx
            prev = a
    #     print(closest_p, closed_p_idx)
        return dist_from_origin
    #         b = [dendrite[0][idx+1],dendrite[1][idx+1]] 
            
    def Is_On_Dendrite(self,dendrite_name,dendrite,point,max_dist):
        """
            function that checks on which segment of the dendrite the point is present (if)
            input: dendrite_name,dendrite,point,max_dist
            output: True/False and scaled distance from the origin of the dendrite
        """
        length_from_origin = 0
        prev_distance = 10**20
        for idx,x in enumerate(dendrite[0][:-1]):
    #         x,y = p
            y = dendrite[1][idx]
    #         print(x,y)
    #     print()
            a = [x,y]
            b = [dendrite[0][idx+1],dendrite[1][idx+1]] 
            if self.isBetween(a,b,point):
                x_int,y_int,distance = self.Perpendicular_Distance_and_POI(a,b,point)
                if distance <= max_dist:
    #                 prev_distance = distance
                    length_from_origin += np.sqrt((y_int-a[1])**2 + (x_int-a[0])**2)
    #                 print("point ",point," belong to dendrite ",dendrite_name," Distance is ", distance, " POI is ",(x_int,y_int))
                    return True, length_from_origin/self.scale
            length_from_origin += np.sqrt((b[1]-a[1])**2 + (b[0]-a[0])**2)
    
        length_from_origin = self.GetClosestRoiPoint(dendrite,point)
        return False, length_from_origin*self.scale
    #set somatic = False for dendritic punctas
    def GetPunctaStats(self,x,y,r,original_img):
        """
            function that claculates the stats of gaussian puncta centered at x,y with radius r
            input: x,y, r and original image called by PunctaDetection class object
            output: list that includes the max, min,mean,std and median of the pixels in circle at x,y with radius r
        """
        #
        img = np.zeros(original_img.shape, dtype=np.uint8)
        rr, cc = disk((y, x), r,shape=original_img.shape)
        img[rr, cc] = 1
        f_img = np.multiply(original_img,img)
        f_img_data = original_img[np.nonzero(f_img)]
        puncta_stats = [f_img_data.max(),f_img_data.min(),f_img_data.mean(),f_img_data.std(),np.median(f_img_data)]
    #     print("punct stat: ",puncta_stats)
        return puncta_stats
    def GetPunctas(self):
        """
            function that does the puncta detection
            input: none, called by PunctaDetection class object
            output: two dictionaries that stores list of puncta stats for each puncta element wise (soma/dendrite)
        """
        all_c_somatic_puncta = {}; 
        all_c_dendritic_puncta = {}
        for ch in self.channels:
            somatic_puncta = {};
            dendritic_puncta = {}
            fig, axes = plt.subplots(1,2, figsize=(20, 10), sharex=True, sharey=True)
            ax = axes.ravel()
        
            orig_img = self.tiff_Arr[0,ch,:,:].astype(float)
            print(orig_img.shape)
            lsm_img = np.zeros(np.shape(orig_img),'uint8')
        
            soma_polygons = {}
            dendrite_lines = {}
            soma_ps = {}
            soma_pxs = []
            soma_img = np.zeros(np.shape(orig_img),'uint8')
            anti_soma = np.ones(np.shape(orig_img),'uint8')
            # breakpoint()
            for soma in self.somas.keys():
                print("soma key = ",soma)
                lsm_img = np.zeros(np.shape(orig_img),'uint8')
                soma_instance = self.somas[soma]
                # breakpoint()
                # soma_polygons[soma] = [soma_instance['x'],soma_instance['y']]
                xs = soma_instance[:,0]
                ys = soma_instance[:,1]
                rr, cc = polygon(ys, xs, lsm_img.shape)
                print(np.shape(rr),np.shape(cc))
                soma_ps[soma] = orig_img[rr,cc] 
                lsm_img[rr,cc] = 1
                anti_soma = np.multiply(anti_soma,1 - lsm_img)
                print()
        #         print(np.quantile(soma_ps[soma],0.75))
        #         print(np.quantile(soma_ps[soma],0.25))
        #         print(np.quantile(soma_ps[soma],0.5))
                t = (np.quantile(soma_ps[soma],0.5))
                soma_img = np.multiply(orig_img,lsm_img)
                blobs_log = blob_log(soma_img,threshold=t)
                blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
                sp = []
                count =0
                for blob in blobs_log:
                    y, x, r = blob
                    i_x = int(x)
                    i_y = int(y)
                    count += 1;
            #         mod_t[i_y][i_x] = t[idx][i_y][i_x]
                    c = plt.Circle((x, y), r, color='k', linewidth=1, fill=False)
                    puncta_stats = [x,y,r]
                    puncta_stats += self.GetPunctaStats(x,y,r,orig_img)
                    puncta_stats += [True,0.0]
                    sp.append(puncta_stats)
                    ax[1].add_patch(c)
                somatic_puncta[soma] = sp
                print(count)#,"\n",len(mod_t))
                ax[0].plot(xs,ys,'w+-')
                ax[1].plot(xs,ys,'w+-')
            ax[0].imshow(orig_img)
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[1].imshow(orig_img)
            plt.savefig(self.Dir+"soma_channel_"+str(ch)+".png")
            plt.savefig(self.Dir+"soma_channel_"+str(ch)+".eps")
        #     plt.show()
            dendrite_img = np.zeros(np.shape(orig_img),'uint8')
            # width = 5*self.scale
            dend_ps = []
            dilated = np.zeros(np.shape(orig_img),'uint8')
            fig, [ax0,ax1] = plt.subplots(1,2, figsize=(20, 10), sharex=True, sharey=True)
            dend_count = 0;
            for jdx,dendrite in enumerate(self.dendrites.keys()):
                if 1:
                    count =0
                    dendrite_instance = self.dendrites[dendrite]
                    # print(dendrite_instance.shape)
                    dendrite_lines[dendrite] = []
                    xs = dendrite_instance[:,0]
                    ys = dendrite_instance[:,1]
                    for lk in range(0,len(xs)-1):
        #             print(dendrite_lines[dendrite][lk][0])
                        rr, cc = line(int(ys[lk]),int(xs[lk]),int(ys[lk+1]),int(xs[lk+1]))
                        dendrite_img[rr,cc] = 1
                          # dendrite_lines[dendrite].append([rr,cc])
                    dilated = morphology.dilation(dendrite_img, morphology.disk(radius=self.width))
                    dilated = np.multiply(anti_soma,dilated)
                    ## uncomment if you don't want to repeat dendritic punctas in overlapping dendritic parts
                    # anti_soma = np.multiply(anti_soma,1 - dilated)
                    dend_img = np.multiply(dilated,orig_img)
                    # print
                    filtered_dend_img = dend_img[np.nonzero(dend_img)]
                    # print(dendrite + " ", np.quantile(filtered_dend_img,0.75))
                    # print(dendrite + " " , np.quantile(filtered_dend_img,0.25))
                    # print(dendrite + " " , np.quantile(filtered_dend_img,0.5))
                    t = (np.quantile(filtered_dend_img,0.75))
                    dend_blobs_log = blob_log(dend_img, threshold=t)
                    # print("ff",dend_blobs_log.shape)
                    dend_blobs_log[:, 2] = dend_blobs_log[:, 2] * sqrt(2)
                    dp = []
                    for blob in dend_blobs_log:
                        # print("plotting circles around puncta")
                        y, x, r = blob
                        i_x = int(x)
                        i_y = int(y)
                        count += 1;
                        dend_count += 1;
                        puncta_stats = [x,y,r]
                        puncta_stats += self.GetPunctaStats(x,y,r,orig_img)
                        on_dendrite, distance_from_origin = self.Is_On_Dendrite(dendrite,[xs,ys],[x,y],self.width)
                        puncta_stats += [on_dendrite, distance_from_origin]
                        dp.append(puncta_stats)
                        c = plt.Circle((x, y), r, color='r', linewidth=.5, fill=False)
                        ax1.add_patch(c)
                    ax0.plot(xs,ys,'w',alpha=0.5)
                    ax1.plot(xs,ys,'w',alpha=0.5)
                    print("found ",count, " mRNAs in ",dendrite)#,"\n",len(mod_t))
                dendritic_puncta[dendrite] = dp
            all_c_somatic_puncta[ch] = somatic_puncta
            all_c_dendritic_puncta[ch] = dendritic_puncta
            ax0.imshow(orig_img)
            ax1.imshow(orig_img)
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            plt.savefig(self.Dir+"dend_channel_"+str(ch)+"_"+str(self.width*self.scale)+".png")
            plt.savefig(self.Dir+"dend_channel_"+str(ch)+"_"+str(self.width*self.scale)+".eps")
            print("total dendritic count = ",dend_count)
            plt.show()
        return all_c_somatic_puncta,all_c_dendritic_puncta
        
        
        
        
        
        
        