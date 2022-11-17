#!/usr/bin/env python
# coding: utf-8

# # **YOLOv5 - Real-Time Social Distancing Detector and People Counter**

# ## **0.** Settings

# In[ ]:


# Install Yolov5
get_ipython().system('git clone https://github.com/ultralytics/yolov5')
get_ipython().run_line_magic('cd', './yolov5')
get_ipython().system('pip install -r requirements.txt')


# In[20]:


from matplotlib import pyplot as plt
import numpy as np
import torch
import math
import cv2


# ## **1.** Model

# In[2]:


# Download the pre-trained YOLOv5 model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# ## **2.** Inspect the video

# In[7]:


# Open the video
cap = cv2.VideoCapture('./Miscellaneous/people.mp4')

while cap.isOpened():

    # Capture the frame
    ret, frame = cap.read()

    # If we correctly captured the frame
    if ret == True:

        # Display the frame
        cv2.imshow('Original video', frame)

        # If we press 'q' then we exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # If we did not capture correctly the frame we exit
    else:
        break

# Close everything in the end
cap.release()
cv2.destroyAllWindows()


# The output is:
# 
# ![Gif](./Miscellaneous/original.gif)

# ## **3.** YOLOv5 detections

# In[8]:


cap = cv2.VideoCapture('./yolov5/data/images/Smart Helmet.mp4')

while cap.isOpened():

    ret, frame = cap.read()

    if ret == True:
        
        # Make detections
        results = model(frame)

        # Display
        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break

cap.release()
cv2.destroyAllWindows()


# The output is:
# 
# ![Gif](./Miscellaneous/yolov5.gif)

# ## **4.** Get the average height of the boxes

# In[11]:


# Display settings
red   = (0,0,255)
green = (0,255,0)
blue  = (255,0,0)
black = (0,0,0)
white = (255, 255, 255)

# Geometric figures settings
thickness = 3
circle_radius = 6
fill = -1 # to fill the geometric figure

# Text settings
text_thickness = 1
text_size = 0.4
title_thickness = 2
title_size = 1
title = 'Real-time social distancing detection system'
font = cv2.FONT_HERSHEY_SIMPLEX # or cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture('./yolov5/data/images/Smart Helmet.mp4')

while cap.isOpened():

    ret, frame = cap.read()
    
    if ret == True:

        # Let's define a variable to save all the heights
        heights = 0

        # Predictions
        results = model(frame)

        # We extract the needed informations: xyxy, xywh
        predictions_xyxy = results.pandas().xyxy[0]
        predictions_xywh = results.pandas().xywh[0]

        # Let us consider only the 'person' label
        predictions_xyxy = predictions_xyxy[predictions_xyxy['name']=='person']
        predictions_xywh = predictions_xywh[predictions_xywh['name']=='person']

        # Let's adjust the indeces (they might be not good since we considered just the 'person' label)
        predictions_xyxy.index = range(len(predictions_xyxy))
        predictions_xywh.index = range(len(predictions_xywh))

        # For every person in the frame:
        for n in range(len(predictions_xyxy)):

            # Let's add-up the height of the box
            heights += predictions_xywh['height'][n]

            # Save the coordinates of the box
            x_min = int(predictions_xyxy['xmin'][n])
            y_min = int(predictions_xyxy['ymin'][n])
            x_max = int(predictions_xyxy['xmax'][n])
            y_max = int(predictions_xyxy['ymax'][n])

            # and the coordinates of the center of each box
            x_center = int(predictions_xywh['xcenter'][n])
            y_center = int(predictions_xywh['ycenter'][n])

            # Let's draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), red, thickness);
            cv2.putText(frame, 'Person', (x_min-3, y_min-5), font, text_size, red, text_thickness);

            # and a blue dot to represent the center of the box
            cv2.circle(frame, (x_center, y_center), circle_radius, blue, fill)

        # Evaluate the average height of the boxes in the current frame
        average_height = heights/len(predictions_xyxy)
        average_height = 'Average height of boxes ' + str(average_height)

        # Print the average height of the boxes
        cv2.putText(frame, average_height, (50,50), font, title_size, white, title_thickness);

        # Show everything: frame, boxes, centers, average height
        cv2.imshow(title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break

cap.release()
cv2.destroyAllWindows()


# The output is:
# 
# ![Gif](./Miscellaneous/box_heights.gif)

# ## **5.** Social distancing detector and people counter

# First, we need to determine what will be the minimal distance to be respected.
# 
# To do this, we need to do some assumptions:
# * we consider that a person is, on average, 1 meter and 70 cm tall
# * we consider that a good distance is, about, 1 meter and 30 cm
# * an (approximate) average of the heights of the boxes is 65
# 
# That said, let's disregard the three-dimensional geometry of the scene for a moment and let's focus instead on the relation:
# 
# $\qquad\qquad average\hspace{2pt}human\hspace{2pt}height\hspace{2pt}:\hspace{2pt}minimal\hspace{2pt}distance\hspace{2pt}in\hspace{2pt}reality\hspace{2pt}=\hspace{2pt} average\hspace{2pt}box\hspace{2pt}height\hspace{2pt}:\hspace{2pt}minimal\hspace{2pt}distance\hspace{2pt}between\hspace{2pt}points$
# 
# So that, we can calculate the $minimal\hspace{2pt}distance\hspace{2pt}between\hspace{2pt}points$ as:
# 
# $\qquad\qquad minimal\hspace{2pt}distance\hspace{2pt}between\hspace{2pt}points = average\hspace{2pt}box\hspace{2pt}height\hspace{2pt}\cdot\hspace{2pt}\frac{minimal\hspace{2pt}distance\hspace{2pt}in\hspace{2pt}reality}{average\hspace{2pt}human\hspace{2pt}height}$
# 

# In[19]:


average_box_height = 65
average_human_height = 170 #cm
minimal_distance_in_reality = 130 #cm
minimal_distance_between_points = (average_box_height * minimal_distance_in_reality)/average_human_height

print('The minimal distance between two centers in the frame has to be: ', minimal_distance_between_points)


# In[22]:


# Display settings
red   = (0,0,255)
green = (0,255,0)
blue  = (255,0,0)
black = (0,0,0)
white = (255, 255, 255)

# Geometric figures settings
thickness = 3
circle_radius = 6
fill = -1 # to fill the geometric figure

# Text settings
text_thickness = 1
text_size = 0.4
title_thickness = 2
title_size = 1
title = 'Real-time social distancing detection system'
font = cv2.FONT_HERSHEY_SIMPLEX # or cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture('./Miscellaneous/people.mp4')

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret == True:

        # Predictions
        results = model(frame)

        # We extract the needed informations: xyxy, xywh
        predictions_xyxy = results.pandas().xyxy[0]
        predictions_xywh = results.pandas().xywh[0]

        # Let us consider only the 'person' label
        predictions_xyxy = predictions_xyxy[predictions_xyxy['name']=='person']
        predictions_xywh = predictions_xywh[predictions_xywh['name']=='person']

        #  Let's adjust the indeces (they might be not good since we considered just the 'person' label)
        predictions_xyxy.index = range(len(predictions_xyxy))
        predictions_xywh.index = range(len(predictions_xywh))

        # In this vector we will save (with 1's) the elements for which we want to make red boxes
        colori_box = [0] * len(predictions_xyxy)
        
        # For every person in the frame:
        for n in range(len(predictions_xyxy)):

                # n-th person's box center coordinates
                x_center = int(predictions_xywh['xcenter'][n])
                y_center = int(predictions_xywh['ycenter'][n])

                # For each person, we create a vector of distances w.r.t. all other people
                # e.g. for person number 0, two vectors will be created:
                #           distances = [5, 5, 2]
                #           distances_indeces = [1, 2, 3]
                #      which means that the person closest to person 0 is person 3

                distances = []
                distances_indeces = []

                for m in range(len(predictions_xyxy)):
                    if m != n:
                        x_center_m = int(predictions_xywh['xcenter'][m])
                        y_center_m = int(predictions_xywh['ycenter'][m])
                        centers_distance = math.dist((x_center, y_center), (x_center_m, y_center_m))
                        distances.append(centers_distance)
                        distances_indeces.append(m)

                # Calculate now the minimum distance (in the above example is 2)
                minimal_distance = np.min(distances)
                # and the index of the minimum distance element (in the example above is 3)
                minimal_distance_element = distances_indeces[np.argmin(distances)]

                # If the two people (the two centers) are too close then both will be assigned flags = 1 in 'colors_box'
                if minimal_distance < minimal_distance_between_points:
                    if colori_box[n] == 0:
                        colori_box[n] = 1
                    if colori_box[minimal_distance_element] == 0:
                        colori_box[minimal_distance_element] = 1

        # Once defined the vector 'colors_box', let's print
        for n in range(len(predictions_xyxy)):
            x_min = int(predictions_xyxy['xmin'][n])
            y_min = int(predictions_xyxy['ymin'][n])
            x_max = int(predictions_xyxy['xmax'][n])
            y_max = int(predictions_xyxy['ymax'][n])

            if colori_box[n] == 1:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), red, thickness);
                cv2.putText(frame, 'Person', (x_min-3, y_min-5), font, text_size, red, text_thickness);
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), green, thickness);
                cv2.putText(frame, 'Person', (x_min-3, y_min-5), font, text_size, green, text_thickness);

            # Also, we always print the center of the box in blue
            x_center = int(predictions_xywh['xcenter'][n])
            y_center = int(predictions_xywh['ycenter'][n])
            cv2.circle(frame, (x_center, y_center), circle_radius, blue, fill)

        # People counter
        people_counter = 'Number of people ' + str(len(predictions_xyxy))
        cv2.putText(frame, people_counter, (50,50), font, title_size, white, title_thickness);

        # Plot all
        cv2.imshow(title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()


# The output is:
# 
# ![Gif](./Miscellaneous/final.gif)
