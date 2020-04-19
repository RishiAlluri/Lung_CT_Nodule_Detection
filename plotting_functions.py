# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:26:14 2020

@author: Ahmed
"""

import pandas as pd
import SimpleITK
import os
import numpy as np
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from PIL import Image
import glob
import helpers
from matplotlib.patches import Circle

def plot(radii,centroids,img_array,path_,chances = None,real = False):
    print(img_array)
    df_cent = pd.DataFrame(centroids,columns = ['X','Y','Z'])
    df_cent['rad'] = radii
    if real:
        df_cent['chances'] = 1
    else:
        df_cent['chances'] = chances
    z_unique = sorted(df_cent['Z'].unique())
    paths = []
    sub_dfs = []
    for i in range(len(z_unique)):
        if real:
            if not os.path.exists('Real\\{}\\{}'.format(path_,i+1)):
                os.mkdir('Real\\{}\\{}'.format(path_,i+1))
        else:
            if not os.path.exists('Predicted\\{}\\{}'.format(path_,i+1)):
                os.mkdir('Predicted\\{}\\{}'.format(path_,i+1))
                
        slice_number = z_unique[i]
        slices = img_array[(int(z_unique[i])-5):(int(z_unique[i])+6)]
        paths_in = []
        #sub_dfs.append(df_cent[df_cent['Z'] == z_unique[i]])
        df_tmp = df_cent[df_cent['Z'] == z_unique[i]]
        df_tmp = df_tmp[df_tmp['chances'] == max(df_tmp['chances'])]
        for k in range(10):
            
            plt.figure(figsize=(30,30))
            plt.imshow(slices[k],cmap=plt.cm.gray)
            #print(len(df_cent[df_cent['Z'] == z_unique[i]]))
            
            #for j in range(len(df_cent[df_cent['Z'] == z_unique[i]])):
                
            plt.axis("off")
            #plt.gca().xaxis.set_major_locator(plt.NullLocator())
            #plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            alphas = [0.1,0.15,0.2,0.25,0.3,0.3,0.25,0.2,0.15,0.1]
            circ = Circle((df_tmp['Y'].values[0],df_tmp['X'].values[0]),2*df_tmp['rad'].values[0],linewidth=0,alpha = alphas[k],color = 'red')
            plt.gcf().gca().add_artist(circ)
            if real:
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig('Real\\{}\\{}\\{}.png'.format(path_,i+1,int(z_unique[i])-5+k), bbox_inches = 'tight',pad_inches = 0)
                paths_in.append('Real\\{}\\{}\\{}.png'.format(path_,i+1,int(z_unique[i])-5+k))
            else:
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig('Predicted\\{}\\{}\\{}.png'.format(path_,i+1,int(z_unique[i])-5+k), bbox_inches = 'tight',pad_inches = 0)
                paths_in.append('Predicted\\{}\\{}\\{}.png'.format(path_,i+1,int(z_unique[i])-5+k))
            plt.show()
            circ.remove()
        sub_dfs.append(df_tmp)
        paths.append(paths_in)
    return paths,sub_dfs

#paths = plot(radii,centroids,img_array)



def gif(filename, array, fps=5, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    array = np.array(array)
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

def get_nodules(img_path):
    
    fcount = 0

    #img_path = "1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231.mhd"

    itk_img = SimpleITK.ReadImage(img_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    #center = np.array([node_x,node_y,node_z])   # nodule center
    #origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    #spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

    df_node = pd.read_csv('C:\\Users\\Ahmed\\Desktop\\CT_Gui\\luna16_nodule_predictions\\predictions10_luna_posnegndsb_v2\\' + img_path.split('/')[-1][:-4] + '.csv')

    m_factor = np.array([img_array.shape[1],img_array.shape[2],img_array.shape[0]])

    mini_df = df_node #get all nodules associate with file
    centroids = []
    radii = []
    chances = []
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 
        for i in range(len(mini_df)):
            #biggest_node = mini_df["diameter_mm"].values[i]   # just using the biggest node
            node_x = mini_df["coord_x"].values[i]
            node_y = mini_df["coord_y"].values[i]
            node_z = mini_df["coord_z"].values[i]
            diam = mini_df["diameter_mm"].values[i]
            chance = mini_df['nodule_chance'].values[i]
            m_factor = np.array([img_array.shape[2],img_array.shape[1],img_array.shape[0]])
            centroid = np.array([node_y,node_x,node_z])
            centroid = centroid*m_factor
            centroids.append(centroid)
            radii.append(diam/2)
            chances.append(chance)
    return radii,centroids,chances,img_array