# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:26:02 2021

@author: suvarna
"""

########################## LIBRARIES & MODULES #############################

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import base64
import pngquant

########################## DEFINE FUNCTIONS ################################
# Function to covert Red background image to Transparent background image
def RedToTransparent(red):
    
    img = red.convert("RGBA")    
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return(img)

# Function to download Image
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Fuction to compress image using PNGQUANT
def compress_by_pngquant(crop_img, mn, mx):
        crop_img.save('check.png')
        path_img = 'check.png'
        pngquant.config(min_quality=mn, max_quality=mx)  # "/usr/bin/pngquant", min_quality=60, max_quality=90
        pngquant.quant_image(path_img)#, override=True, delete=True)
        return(path_img)

############################################################################
########################## CODE BEGINS HERE ################################
############################################################################

# Give a title
st.title('TRANSPARENT BACKGROUND IMAGE GENERATOR')

# Upload the images
st.markdown('**White Background Image**')
img_data = st.file_uploader(label='Load White Background Image', type=['png', 'jpg', 'jpeg'])

if img_data is not None:
    
    # Display uploaded image
    uploaded_img = Image.open(img_data)
    st.title('Image with White Background')
    st.image(uploaded_img)

    img = np.array(uploaded_img)
    #img = Image.fromarray(uploaded_img)
    #img = cv2.imread(img_data)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Contour Mapping
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #Masking
    mask = np.zeros(thresh.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    # Draw contours
    masked = cv2.drawContours(mask, [largest_areas[-1]],0,(255, 255, 255, 255),-1)
    

    
    # GRABCUT the image from the mask
    
    img11 = np.asarray(uploaded_img)
    img1 = cv2.cvtColor(img11, cv2.COLOR_RGBA2RGB)
    newmask = masked
    
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    mask_ch = np.zeros(img.shape[:2],np.uint8)
    mask_ch[newmask == 0] = 0
    mask_ch[newmask == 255] = 1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    cv2.grabCut(img1,mask_ch,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    #cv2.grabCut(img1,mask_ch,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask1 = np.where((mask_ch==2)|(mask_ch==0),0,1).astype('uint8')
    img1 = img1*mask1[:,:,np.newaxis]
    
    # Converting BGR To RGB and saving
    #final = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    final = img1
    black_image = Image.fromarray(final.astype(np.uint8))
    
    # Creating a Red background
    bckgrnd = np.full_like(final, (255, 0, 0), dtype=np.uint8)
    
    mask = masked
    
    # apply inverse mask to colored background image
    bckgrnd_masked = cv2.bitwise_or(bckgrnd, bckgrnd, mask=255-mask)
    
    # combine the two
    result = cv2.add(final, bckgrnd_masked)
    result_image = Image.fromarray(result.astype(np.uint8))
    
    #Generate output image
    # Converting Red to transparent background and saving result
    out_image = RedToTransparent(result_image)
    
    #Getting size of image
    size_file = BytesIO()
    out_image.save(size_file, 'png')
    out_size = size_file.tell()/1000000
    
    #display size
    st.markdown('**Size of Transparent Background Image**') 
    #size
    st.text('Size: ' + str(out_size) + ' MB,' + '  Dimensions: ' + str(out_image.size))   
    
    
    ######## Cropping Image ###########
    # Find the bounding box from largest contour
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    l,t,r,b = x,y,x+w,y+h
    
    # Cropping
    out1 = out_image
    crop_image = out1.crop((l, t, r, b))
    imgch = crop_image
    
    #Getting size of image
    size_file = BytesIO()
    imgch.save(size_file, 'png')
    out_size = size_file.tell()/1000000
    
    #display size
    st.markdown('**Size of Cropped Image with Transparent Background**')
    #size
    st.text('Size: ' + str(out_size) + ' MB,' + '  Dimensions: ' + str(imgch.size)) 
    
    ######### Resizing Image #############
    fixed_height = 420
    
    if(imgch.size[1] > fixed_height):
        height_percent = (fixed_height / float(imgch.size[1]))
        width_size = int((float(imgch.size[0]) * float(height_percent)))
        resized = imgch.resize((width_size, fixed_height), Image.ANTIALIAS)
        
    else:
        resized = imgch
        
    #Getting size of image
    size_file = BytesIO()
    resized.save(size_file, 'png')
    out_size = size_file.tell()/1000000
    
    #display Resized image and size
    st.title('Resized Image with Transparent Background')
    st.image(resized) 
    #size
    st.text('Size: ' + str(out_size) + ' MB,' + '  Dimensions: ' + str(resized.size)) 
    
    ## Original image came from cv2 format, fromarray convert into PIL format
    img_file2 = 'Resized.png'
    st.markdown(get_image_download_link(resized,img_file2,'Download '+img_file2), unsafe_allow_html=True)

    
    ######### Compressing Image-PNGQUANT #############
    st.title('Compressed Image(PNGQuant) with Transparent Background')
    # Select the copression quality
    option = st.selectbox('Please select quality of compression:',
                          ('90-95', '85-95', '75-95', '65-95'))
    
    st.write('You selected:', option)
    
    if(option == '90-95'):
        mnm = 90
        mxm = 95
    elif(option == '85-95'):
        mnm = 85
        mxm = 95
    elif(option == '75-95'):
        mnm = 75
        mxm = 95
    else:
        mnm = 65
        mxm = 95
        
    comp_img = compress_by_pngquant(crop_image, mnm, mxm)
    compressed = Image.open(comp_img)
    
    #Getting size of image
    size_file = BytesIO()
    compressed.save(size_file, 'png')
    out_size = size_file.tell()/1000000
    
    #display Compressed image and size
    
    st.image(comp_img) 
    #size
    st.text('Size: ' + str(out_size) + ' MB,' + '  Dimensions: ' + str(compressed.size)) 
    
    ## Original image came from cv2 format, fromarray convert into PIL format
    img_file2 = 'Compressed.png'
    st.markdown(get_image_download_link(compressed,img_file2,'Download '+img_file2), unsafe_allow_html=True)
    
