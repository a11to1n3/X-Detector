import os
import copy
import pandas as pd
import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms, utils, models
from sklearn.cluster import DBSCAN
from matplotlib import cm
import matplotlib.pyplot as plt

# Erase classes except one class
def eraseWithROI(image, start_point, end_point):
  color = (int(image[start_point[1], start_point[0],0]),int(image[start_point[1], start_point[0],1]),int(image[start_point[1], start_point[0],2]))
  image = cv.rectangle(image, start_point, end_point, color , -1)

  return image

# multiplying instance
def multiply(image, instance, knowledge_base, times):
  for t in range(times):
    while 1:
      for k in knowledge_base:
        exit_loop = False
        position = (np.random.randint(image.shape[0]), np.random.randint(image.shape[1]))
        if (len(list(set(range(position[0],instance.shape[0] + position[0])) & set(range(k[0],k[1])))) != 0 \
          and (len(list(set(range(position[1],instance.shape[1] + position[1])) & set(range(k[2],k[3])))) != 0)) \
          or (instance.shape[0] + position[0]) > image.shape[0] \
          or (instance.shape[1] + position[1]) > image.shape[1]:
          break
        exit_loop = True

      if exit_loop:
        break

    knowledge_base.append((position[0], instance.shape[0] + position[0], position[1], instance.shape[1] + position[1]))
    image[position[0]:instance.shape[0] + position[0],position[1]:instance.shape[1] + position[1]] = instance
  
  return image

# Multiplying instance
def multiplyAndWrite(image, instance, knowledge_base, times, class_of_interest, count):
  for t in range(times):
    while 1:
      for k in knowledge_base:
        exit_loop = False
        position = (np.random.randint(image.shape[0]), np.random.randint(image.shape[1]))
        if (len(list(set(range(position[0],instance.shape[0] + position[0])) & set(range(k[0],k[1])))) != 0 \
          and (len(list(set(range(position[1],instance.shape[1] + position[1])) & set(range(k[2],k[3])))) != 0)) \
          or (instance.shape[0] + position[0]) > image.shape[0] \
          or (instance.shape[1] + position[1]) > image.shape[1]:
          break
        exit_loop = True

      if exit_loop:
        break

    knowledge_base.append((position[0], instance.shape[0] + position[0], position[1], instance.shape[1] + position[1]))
    image[position[0]:instance.shape[0] + position[0],position[1]:instance.shape[1] + position[1]] = instance
    cv.imwrite(f'./augmented_img/{class_of_interest}_{str(count)}.bmp', image)
    count += 1

    for k in knowledge_base:
      # Flip horizontally
      start_point = (k[0],k[2])
      end_point = (k[1], k[3])
      image_copy = copy.deepcopy(image)
      instance1 = cv.flip(image_copy[start_point[0]:end_point[0], start_point[1]:end_point[1]], 1)
      image_copy[start_point[0]:end_point[0], start_point[1]:end_point[1]] = instance1
      cv.imwrite(f'./augmented_img/{class_of_interest}_{str(count)}.bmp', image_copy)
      count += 1

      # Flip vertically
      image_copy = copy.deepcopy(image)
      instance1 = cv.flip(image_copy[start_point[0]:end_point[0], start_point[1]:end_point[1]], 0)
      image_copy[start_point[0]:end_point[0], start_point[1]:end_point[1]] = instance1
      cv.imwrite(f'./augmented_img/{class_of_interest}_{str(count)}.bmp', image_copy)
      count += 1
  
  return image, knowledge_base, count

def find_bounding_boxes_per_class(CAM_explainer, img_path, class_names, class_labels, class_colors, plot=False):
    strToLabel = {n:l for n,l in zip(class_names, class_labels)}
    strToColor = {n:c for n,c in zip(class_names, class_colors)}

    if plot:
        fig = plt.figure(figsize=(20,70))
        ax = fig.gca()

    class_boxes = {}
    class_scores = {}
    for class_oi in list(strToLabel.keys()):
        class_boxes[class_oi] = []
        class_scores[class_oi] = []
    for class_oi in list(strToLabel.keys()):
        img = cv.imread(img_path)
        data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        img_norm = data_transform(img)
        ex, out = CAM_explainer(torch.tensor(img_norm.reshape(1,3,224,224),dtype=torch.float),False,strToLabel[class_oi])
        
        cmap = cm.get_cmap('jet')
        ex = Image.fromarray(ex)
        img = Image.fromarray(img)
        overlay = ex.resize(img.size, resample=Image.BILINEAR)
        overlay = np.asarray(overlay)

        overlay1 = overlay>0.8
        X = []
        for i in range(overlay1.shape[1]):
            for j in range(overlay1.shape[0]):
                if overlay1.T[i,j] == 1:
                    X.append([i,j])

        X = np.array(X)
        if len(X.shape) > 1:
            clustering = DBSCAN(eps=3, min_samples=2).fit(X)
        else:
            if len(X) == 1:
                clustering = DBSCAN(eps=3, min_samples=2).fit(X.reshape(1, -1))
            else:
                clustering = DBSCAN(eps=3, min_samples=2).fit(X.reshape(-1, 1))

        xmins = [min(np.array(X)[clustering.labels_ == l,0]) for l in np.unique(clustering.labels_) if l != -1]
        xmaxs = [max(np.array(X)[clustering.labels_ == l,0]) for l in np.unique(clustering.labels_) if l != -1]
        ymins = [min(np.array(X)[clustering.labels_ == l,1]) for l in np.unique(clustering.labels_) if l != -1]
        ymaxs = [max(np.array(X)[clustering.labels_ == l,1]) for l in np.unique(clustering.labels_) if l != -1]

        [class_boxes[class_oi].append([xmins[i], ymins[i], xmaxs[i], ymaxs[i]]) for i in range(len(xmins))]
        [class_scores[class_oi].append(np.mean(overlay[class_boxes[class_oi][i][0]:class_boxes[class_oi][i][2],class_boxes[class_oi][i][1]:class_boxes[class_oi][i][3]])) for i in range(len(xmins))]

        if plot:
            overlay = overlay * (overlay > 0.5).astype(np.uint8)
            overlay = (255 * cmap(overlay ** 2)[:, :, :3]).astype(np.uint8)
            try:
                overlayed_img = Image.fromarray((0.95 * np.asarray(overlayed_img) + (1 - 0.95) * overlay).astype(np.uint8))
            except:
                overlayed_img = Image.fromarray((0.7 * np.asarray(img) + (1 - 0.7) * overlay).astype(np.uint8))
            draw = ImageDraw.Draw(overlayed_img)
            [draw.rectangle(((xmins[i], ymins[i]), (xmaxs[i], ymaxs[i])),outline=strToColor[class_oi]) for i in range(len(xmins))]
            if np.all(np.array(ymins)-10 > 10):
                [draw.text(((xmins[i], ymins[i]-10)),class_oi, fill=strToColor[class_oi]) for i in range(len(xmins))]
            else:
                [draw.text(((xmins[i], ymaxs[i]+10)),class_oi, fill=strToColor[class_oi]) for i in range(len(xmins))]
            
            ax.imshow(np.array(overlayed_img))
    
    if plot:
        plt.show()

    return class_boxes, class_scores
