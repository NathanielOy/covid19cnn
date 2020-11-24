# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:22:32 2020

@author: Oyelade
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from google.colab import drive
import tensorflow as tf
from skimage.transform import resize
import glob
import imageio
import pandas as pd 
from csv import reader
import json
import re
from collections import Counter
#from google.colab.patches import cv2_imshow
# Accessing My Google Drive
#drive.mount('/content/drive')

content="/content/"
myfile="/content/metadata.csv"
myfile2="/content/drive/My Drive/CovidNet/dataset/data/Data_Entry_2017.csv"
bboxfile="/content/drive/My Drive/CovidNet/dataset/tmp/BBox_List_2017.csv"
myfile3="/content/Data_Entry_2017.csv"
content2="/content/drive/My Drive/CovidNet/dataset/data/"
mydir="/content/images/"
base_dir="/content/drive/My Drive/CovidNet/dataset"
# defining global variable path
# Location of my dataset on My Google Drive
image_path_raw = "/content/drive/My Drive/CovidNet/dataset/raw/"
image_path_processed = "/content/drive/My Drive/CovidNet/dataset/processed/"
image_path_segmented = "/content/drive/My Drive/CovidNet/dataset/segmented/"
dest_raw = image_path_raw + 'train/'
dest_raw2 = image_path_raw + 'train2/'
tmp_dest_raw2 = image_path_raw + 'tmpraw4/'
dest_processed = image_path_processed + 'train/'
dest_processed_val = image_path_processed + 'val/'
dest_segmented = image_path_segmented + 'train/'

def loadImages(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    imgformat = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    image_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path)
                          if file.endswith(tuple(imgformat))])
    return image_files


# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display two images
def display2(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), cv2_imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), cv2_imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()
    
    
# Preprocessing
def processing(data):
    # Reading all images to work
    #img = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in data]  #IMREAD_GRAYSCALE, COLOR_BGR2RGB
    '''    
    try:
        print('Original size',img[0].shape)
    except AttributeError:
        print("shape not found")
    '''
    #display_one(img[1])
    # --------------------------------
    # setting dim of the resize
    height = 220
    width = 220
    dim = (width, height)
    findings=[]
    
    #res_img = []
    no_noise = []
    for img_file in data:
        #9=img75001_Effusion.png,  8=img65001_Mass.png,  7=img55001_NoFinding.png,  6=img45000_NoFinding.png        
        #5=img35000_Infiltration.png,  #4=img25000_NoFinding.png
        #img_file='/content/drive/My Drive/CovidNet/dataset/processed/test/img35000_NoFinding.png'
        real_img_file=img_file
        pme=img_file.split('/')
        idx=len(pme)-1
        img_file=pme[idx]
        digit_me=re.findall(r'\d+', img_file)        
        if int(digit_me[0]) >= 75001:
            print((digit_me[0])+'  '+img_file)            
            img=cv2.imread(real_img_file, cv2.IMREAD_GRAYSCALE)
            findings.append(real_img_file)
            res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            blur = cv2.GaussianBlur(res, (5, 5), 0)#.astype('uint8')
            no_noise.append(blur)
            #res_img.append(res)
        

    '''
    # Checcking the size
    try:
        print('RESIZED', res_img[1].shape)
    except AttributeError:
        print("shape not found")
    '''
    
    # Visualizing one of the images in the array
    #original = res_img[1]
    #display_one(original)
    # ----------------------------------
    # Remove noise
    # Using Gaussian Blur
    '''
    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)#.astype('uint8')
        no_noise.append(blur)
    '''

    #image = no_noise[1]
    #display(original, image, 'Original', 'Blured: No Noise')
    #---------------------------------
    # Segmentation
    
    segmented_imgs = []
    sure_foreground_area_imgs = []
    '''
    real_process_imgs = []
    for i in range(len(no_noise)):
        gray=no_noise[i]
        ret, seg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Further noise removal (Morphology)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel, iterations=2)
    
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
    
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
    
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
    
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        sure_foreground_area_imgs.append(sure_bg)
        
        segmented_imgs.append(seg)
        real_process_imgs.append(gray)
      
    
    # Displaying segmented images
    thresh=segmented_imgs[1]
    display(original, thresh, 'Original', 'Segmented')

    #Displaying segmented back ground
    sure_bg=sure_foreground_area_imgs[1]
    display(original, sure_bg, 'Original', 'Segmented Background')
    '''
    return no_noise, findings, segmented_imgs #real_process_imgs, findings, segmented_imgs 

    '''
    Code below only works for input 3 channels (RGB images)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    # Displaying markers on the image
    display(original, markers, 'Original', 'Marked')
    '''

def write2process_store(real_process_imgs, findings, segmented_imgs):
    path_process=image_path_processed
    path_segmented=dest_segmented
    
    for i in range(len(real_process_imgs)):
        #imageio.imwrite(path_process, real_process_imgs[i])
        #imageio.imwrite(path_segmented, segmented_imgs[i])
        with open(dest_raw2+'/NHD_train_list.txt') as fileme1:
            train = [line.rstrip('\n') for line in fileme1]
        with open(dest_raw2+'/NHD_val_list.txt') as fileme2:
            val = [line.rstrip('\n') for line in fileme2]
        with open(dest_raw2+'/NHD_test_list.txt') as fileme3:
            test = [line.rstrip('\n') for line in fileme3]
        
        pme=findings[i].split('/')
        idx=len(pme)-1
        img=pme[idx].split('_')
        tmp=img[1].split('.')
        finding=tmp[0]
        if pme[idx] in train:
            finding_dir=path_process+'train/'+finding+'/'
        elif pme[idx] in val:
            finding_dir=path_process+'val/'+finding+'/'
        else:
            finding_dir=path_process+'test/'+finding+'/'
        
        ptg=[finding, pme[idx], finding_dir]
        print(ptg)
        
        
        if not os.path.exists(finding_dir):
            os.makedirs(finding_dir)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(finding_dir+'/img_nih_'+str(i)+'_'+finding+'.png', real_process_imgs[i])
        #cv2.imwrite(path_segmented+'/seg_img_nih_'+str(i)+'.png', segmented_imgs[i])
    
def bounding_box(img_file):
    # read and scale down image
    # wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
    # wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
    img = cv2.pyrDown(cv2.imread(img_file, cv2.IMREAD_UNCHANGED))
    
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220, 220, cv2.THRESH_BINARY)
    # find contours and get the external one
    
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
    
        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    
    #print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    #cv2_imshow(img) #cv2.imshow("contours", img)
    '''
    while True:
        key = cv2.waitKey(1)
        if key == 27: #ESC key to break
            break
    '''
    return img
    #cv2.destroyAllWindows()

def bounding_box_2():
    bboxfile='/content/annotations/imageannotation_ai_lung_bounding_boxes.json'
    boxed_imgs=[]
    with open(bboxfile, 'r') as read_obj:
        json_data = json.load(read_obj)
        index=0
        for row in json_data['annotations']:
            if index < 700:
                image_id=row['image_id']
                filepath=''
                for r in json_data['images']:
                    if r['id'] == image_id:
                        filepath=r['file_name']
                        break;
            
                finds=row['attributes']['Finding'][0]
                view=row['attributes']['View'][0]
                path='/content/images/'
                
                found=False
                for item in boxed_imgs:
                    imageid=item[1]
                    if image_id == imageid:
                        found=True
                        path=item[0]
                        #print(path+' Set to true '+str(image_id)+'  '+str(image_id))
                        
                        
                if not found:
                    path=path+filepath
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                finds=finds+' '+view
                x_min=row['bbox'][0]
                y_min=row['bbox'][1]
                x_max=row['bbox'][2]
                y_max=row['bbox'][3]
                cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2) # add rectangle to image
                labelSize=cv2.getTextSize(finds,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
                # print('labelSize>>',labelSize)
                _x1 = x_min
                _y1 = y_min#+int(labelSize[0][1]/2)
                _x2 = _x1+labelSize[0][0]
                _y2 = y_min-int(labelSize[0][1])
                cv2.rectangle(image,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
                cv2.putText(image,finds,(x_min,y_min),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)  
                #print(' path='+path+' x_min='+str(x_min)+' y_min='+str(y_min)+'  finds='+finds+'  view='+view)
                imge=bounding_box(path)
                height = 220
                width = 220
                dim = (width, height)
                res1 = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
                
                cv2_imshow(res1)
                cv2_imshow(res2)
                print(index+'>>> '+finds+': Only ROI '+'- with Contours')
                #display2(res1, res2, finds+': Only ROI', finds+': with Contours')                          
                
                imgbox=[path, image_id]
                boxed_imgs.append(imgbox)
            index+=1
            
            
def prefetch_images():            
    tmpstore=[]
    dfolder=4 
    # open file in read mode         
    isContinue= False
    with open(myfile2, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        index=0
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            #patientid=0, sex=2, age=3, finding=4 survival=5 view=17, modality=18, folder=21, filename=22
            #index=70564
            #9=00018387_035.png, 8=00016051_010.png, 7=00013774_026.png, 6=00011558_008.png, 5=00009232_004.png, 4=00006585_007.png, 3=00003923_014.png, 10=
            if row[0]== '00006585_007.png' or isContinue: #and index < 5: index > 0
                  #patientid=row[0], sex=row[2], age=row[3], folder=row[21], survival=row[5], view=row[17], modality=row[18]
                  finding=row[1]     #finding=row[4]
                  filename=row[0]   #filename=row[22]
                  isContinue= True
                  #print(row[0]+' '+row[2]+' '+row[3]+' '+row[4]+' '+row[5]+' '+row[17]+' '+row[18]+' '+row[21]+' '+row[22])
                  filepath=filename  #mydir+filename
                  if not filename.endswith('.gz'):
                      i=dfolder;
                      while i <= dfolder:
                          number_str = str(i)
                          zero_filled_number = number_str.zfill(3)
                          path=content2+'images_'+str(zero_filled_number)+'/images/'
                          i+=1
                          img_file_path=path+filepath
                          if os.path.exists(img_file_path):
                              #image = cv2.imread(path+filepath)
                              #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                              finding=re.sub("\s+", "", finding.strip())
                              tmpf=finding.split('|')   #finding.split(',')
                              for tsep in tmpf:
                                  finding=tsep
                                  finding=finding.replace("_", "")
                                  finding=finding.replace(".", "")
                                  if 'COVID' in finding:
                                      finding='COVID-19'
                                  filename='img'+str(index)+'_'+finding+'.png'
                                  #cv2.imwrite(tmp_dest_raw2+'/'+filename, image_gray)
                                  tmpstore.append(filename)
                              print('images_'+str(zero_filled_number)+'/images/'+ ' see the image '+filepath+' == '+filename+'  sep >>> '+finding)
                              break
                          else:
                              print('No image')
                      
                      
                      #if index < 2:
                          #original = image
                          #display_one(original, filename)                         
            
                  
            index+=1
            
    
    trainf = open(dest_raw2+'NHD_train_list.txt', 'a') #open(content+'covid_train_list.txt', 'w')
    valf = open(dest_raw2+'NHD_val_list.txt', 'a')     #open(content+'covid_val_list.txt', 'w')
    testf = open(dest_raw2+'NHD_test_list.txt', 'a')   #open(content+'covid_test_list.txt', 'w')
    
    tsize=int((len(tmpstore)*75)//100)
    vsize=int((len(tmpstore)*20)//100)
    #ysize=int((len(tmpstore)*10)//100)
    idx=0
    for line in tmpstore:
        if idx < tsize:
          trainf.write("%s\n" % line)
        elif idx > tsize and idx <= (tsize+vsize):
          valf.write("%s\n" % line)
        else:
          testf.write("%s\n" % line)
        idx+=1

    trainf.close()
    valf.close()

def count_elements(seq) -> dict:
     """Tally elements from `seq`."""
     hist = {}
     for i in seq:
       hist[i] = hist.get(i, 0) + 1
     return hist

def visualize(datainfo, color='#0504aa', facecolor='orangered', edgecolor='maroon'):  #peru, blue
    for dataset in datainfo:
        print("Values of d:", dataset) 
        print("Type of d:", type(dataset)) 
        print("Size of d:", len(dataset))
        plt.bar(range(len(dataset)), list(dataset.values()), align='center')
        plt.xticks(range(len(dataset)), list(dataset.keys()), rotation = 90)
        plt.show();

    
def chart_major_datasets():
    #datasets=['metadata.csv', 'Data_Entry_2017.csv']
    datasetsname=['COVID-19 chest xray', 'NIH Chest X-rays']
    zippeddataset= {}
    total_in_datasets=[]
    findings_in_datasets=[]
    all_train=[]
    all_val=[]
    all_test=[]
    
    with open(content+'/covid_train_list.txt') as fileme1:
        train1 = [line.rstrip('\n') for line in fileme1]
    with open(content+'/covid_val_list.txt') as fileme2:
        val1 = [line.rstrip('\n') for line in fileme2]
    with open(content+'/covid_test_list.txt') as fileme3:
        test1 = [line.rstrip('\n') for line in fileme3]
        
    
    with open(dest_raw2+'NHD_train_list.txt') as fileme4:
        train2 = [line.rstrip('\n') for line in fileme4]
    with open(dest_raw2+'NHD_val_list.txt') as fileme5:
        val2 = [line.rstrip('\n') for line in fileme5]
    with open(dest_raw2+'NHD_test_list.txt') as fileme5:
        test2 = [line.rstrip('\n') for line in fileme5]
        
    train = []
    train.extend(train1)
    train.extend(train2)
    print("Size of train:", len(train))

    val = []
    val.extend(val1)
    val.extend(val2)
    print("Size of val:", len(val))
    
    test = []
    test.extend(test1)
    test.extend(test2)
    print("Size of test:", len(test))
    
    dset1=[]
    dset1.extend(train1)
    dset1.extend(val1)
    dset1.extend(test1)
    dset2=[]
    dset2.extend(train2)
    dset2.extend(val2)
    dset2.extend(test2)
    mydatasets=[dset1, dset2]
    
    for idx, item in enumerate(mydatasets): 
        countitems=0
        findings=[]
        for pme in item: 
            img=pme.split('_')
            tmp=img[1].split('.')
            finding=tmp[0]
            image=pme
            if image in train:
                all_train.append([image, finding])
            elif image in val:
                all_val.append([image, finding])
            else:
                all_test.append([image, finding])
                    
            findings.append(finding)
            countitems+=1
        total_in_datasets.append(countitems)
        findings_in_datasets.append(findings)
        name=datasetsname[idx]
        zippeddataset [name] =countitems        
    
    viewdata=[]
    viewdata.append(zippeddataset)
    visualize(viewdata)
    return findings_in_datasets, all_train, all_val, all_test
    
def chart_classes_of_diseases_datasets(findings_in_d):
    count_elements=[]
    for dataset_f in findings_in_d: 
        count_elements.append(Counter(dataset_f))
    
    visualize(count_elements)
    
def chart_train_eval_test_datasets(all_train, all_val, all_test):
    datasplit={}
    datasplit['Training']=len(all_train)
    datasplit['Testing']=len(all_test)
    datasplit['Validation']=len(all_val)
    
    visualize([datasplit])
    
    elements_train=[]
    elements_val=[]
    elements_test=[]
    for item in all_train: 
        elements_train.append(item[1])
    train=Counter(elements_train)
    for item in all_val: 
        elements_val.append(item[1])
    val=Counter(elements_val)
    for item in all_test: 
        elements_test.append(item[1])
    test=Counter(elements_test)
    
    visualize([train, test, val])
    
    
def main():
    # calling global variable
    global dest_raw2 #dest_raw2
    
    '''
    findings_in_datasets, all_train, all_val, all_test=chart_major_datasets()
    chart_classes_of_diseases_datasets(findings_in_datasets)
    chart_train_eval_test_datasets(all_train, all_val, all_test)
    '''
    
    #bounding_box_2()
    
    #prefetch_images()
    
    
    #The var Dataset is a list with all images in the folder 
    dataset = loadImages(dest_raw2)
    print('number of FILES in dir', len(dataset))
    print("--------------------------------")
    #print(cv2.imread(dataset[0]).shape)
    #print("List of all files in the folder:\n",dataset)
    print("--------------------------------")
    
    # sending all the images to pre-processing
    real_process_imgs, findings, segmented_imgs=processing(dataset)
    print("Done processing...")
    write2process_store(real_process_imgs, findings, segmented_imgs)
    print("Done writing...")
    #list files in directory
    #a = tf.gfile.ListDirectory('drive/My Drive/BiSeNet/dataset/train')
    #print(a)
    

main()    


'''
File contents: Image format: 112,120 total images with size 1024 x 1024,  
images_001.zip: Contains 4999 images, images_002.zip: Contains 10,000 images,  
images_003.zip: Contains 10,000 images,  images_004.zip: Contains 10,000 images, 
images_005.zip: Contains 10,000 images, images_006.zip: Contains 10,000 images,  
images_007.zip: Contains 10,000 images,  images_008.zip: Contains 10,000 images, 
images_009.zip: Contains 10,000 images, images_010.zip: 
Contains 10,000 images, images_011.zip: Contains 10,000 images, images_012.zip: Contains 7,121 images
'''
 