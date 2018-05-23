#!/usr/bin/env python

# General python imports
import numpy as np
import matplotlib.pyplot as plt

# ROS imports
import rospy
from sensor_msgs.msg import Image

# OpenCV2 imports
import cv2
from cv_bridge import CvBridge

# Define global variables
bridge = None
img_rgb = 0
img_depth = 0

minLineLength=200

def callback_get_RGB(data):
    global bridge, img_rgb

    rospy.loginfo("RGB image received!")
    # Convert to CV2 image
    img_rgb = bridge.imgmsg_to_cv2(data,data.encoding )

def callback_get_depth(data):
    global bridge, img_depth

    rospy.loginfo("Depth image received!")
    # Convert to CV2 image
    cv_data = bridge.imgmsg_to_cv2(data, data.encoding)
    # only use real part of first image channel
    img_depth = cv_data

def show_RGB_image():
    global img_rgb

    # Show image in figure 1
    plt.figure(1)
    plt.imshow(img_rgb)

def show_depth_image():
    global img_depth

    ## Convert from distance to float
    # Copy img_depth to local variable
    depth = img_depth * 1.
    # Normalize with maximum distance 10
    depth /= 10.
    # Limit to range [0, 1]
    depth = np.clip(depth, 0., 1.)

    # Shoyprw image in figure 2
    plt.figure(2)
    plt.imshow(depth, cmap=plt.get_cmap('gray'))

def convDepthToArray(depth)  :  
    depth = img_depth * 1.
    rows = np.size(depth,0)
    cols = np.size(depth,1)
    for i in range(rows):
	for j in range(cols):
	    if depth[i, j] >1.:
                depth[i, j] = 1.  
            elif depth[i, j] < 0.:
                depth[i, j] = 0.
    gray = np.uint8(depth*255)
    
    return(gray)
 
    
def rgb_line_detection(img):
    gray=convDepthToArray(img)
    
    imag2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
 
    
    edges = cv2.Canny(gray,50,200)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80, minLineLength = 200, maxLineGap =50)
    lines1 = lines[:,0,:]             
    for x1,y1,x2,y2 in lines1[:]:
        if y2-y1< minLineLength :
            if x2-x1<10 :
                cv2.line(imag2,(x1,y1),(x2,y2),(255,0,0),10)
		
    plt.subplot(122),plt.imshow(img2,)
    plt.xticks([]),plt.yticks([])	    
    

    
def ConvDepthToArray(depth)  :  
    depth = img_depth * 1.
    rows, cols = depth.shape
    
    for i in range(rows):
	for j in range(cols):
	    if depth[i, j] >1.:
                depth[i, j] = 1.  
            elif depth[i, j] < 0.:
                depth[i, j] = 0.
    gray = np.uint8(depth*255)
    
    return(gray)
 
def Depth_line_detection(depth):
    gray=convDepthToArray(depth)
   
   
   # plt.imshow(gray)
    imag2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(gray,50,200)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,80, ) 
    lines1 = lines[:,0,:] 
    for x1,y1,x2,y2 in lines1[:]:
        if y2-y1< minLineLength :
            if x2-x1<10 :
                cv2.line(imag2,(x1,y1),(x2,y2),(255,0,0),1)

		
    
            
            
    plt.subplot(122),plt.imshow(imag2,)
    plt.xticks([]),plt.yticks([])
    
if __name__ == '__main__':

    rospy.init_node('landmark_mapping')

    # Initialize CV bridge
    bridge = CvBridge()

    # Create subscribers
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_get_RGB)
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_get_depth)

    # Idle
    rospy.spin()
