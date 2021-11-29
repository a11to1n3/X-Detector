import os
import copy
import pandas as pd
import numpy as np

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
