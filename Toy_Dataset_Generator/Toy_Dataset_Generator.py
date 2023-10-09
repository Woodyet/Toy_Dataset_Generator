#file to make a dataset for toy example using VQGAN
from PIL import Image
import matplotlib.pyplot as plt

from skimage.draw import random_shapes
from skimage.util import random_noise

import numpy as np
import os
import cv2
import copy

from tqdm import tqdm

def noise_add(img_in,noise):
   x,y = img_in.shape
   if noise == "gauss":
      gauss_noise=np.zeros((x,y),dtype=np.uint8)
      cv2.randn(gauss_noise,128,20)
      gauss_noise=(gauss_noise*0.5).astype(np.uint8)
      return cv2.add(img_in,gauss_noise)
   elif noise == "S&P":
      noise = np.random.randint(0, 2, size=img_in.shape)
      noisy_img = img_in.copy()
      noisy_img[noise == 0] = 0
      noisy_img[noise == 1] = 255
      return noisy_img
   elif noise == "poisson":
      noise = np.random.poisson(img_in)
      noisy_img = np.clip(noise, 0, 255).astype(np.uint8)
      return noisy_img
   elif noise == "random":      
      noise = np.random.random(img_in.shape) * 255
      return img_in + noise

def make_image_n_mask(x,y,use_background_noise,fg_noise_type,bg_noise_type,shape,max_shapes=1):
   result = random_shapes((x, y), max_shapes=max_shapes,shape=shape,channel_axis=-1,num_channels=1)

   image_mask, labels = result
   image_mask = image_mask.reshape(x,y)

   image_mask_final = copy.deepcopy(image_mask)
   image_mask_final[image_mask_final <= 254] = 1
   image_mask_final[image_mask_final == 255] = 0

   blank_image = np.ones((x,y))*127
   blank_image = blank_image.astype(np.uint8) 

   noisy_image = noise_add(blank_image,fg_noise_type)

   if use_background_noise:
       image_mask_final_inv = copy.deepcopy(image_mask)
       image_mask_final_inv[image_mask_final_inv <= 254] = 0
       image_mask_final_inv[image_mask_final_inv == 255] = 1

       bg_noisy_image = noise_add(blank_image,bg_noise_type)
       bg_noisy_image = image_mask_final_inv * bg_noisy_image

       fg_noisy_image = image_mask_final*noisy_image

       final_noisy_image_fin = bg_noisy_image + fg_noisy_image
   else:
       final_noisy_image_fin = image_mask_final*noisy_image

   return image_mask_final,final_noisy_image_fin.astype(np.uint8)

if __name__ == "__main__":
    if sys.platform.startswith("linux"): 
        plat = "linux"
    elif sys.platform == "darwin":
        print("Mac Currently Unsupported")
        sys.exit()
    elif os.name == "nt":
        plat = "win"


    shapes='rectangle', 'circle', 'triangle', 'ellipse',None
    background_noise = True
    fg_noise_types = ['gauss','poisson','random']
    bg_noise_types = ['S&P']

    x=1080
    y=1080

    examples_per_class = 10
    max_num_of_shapes = 10

    #make dirs

    os.mkdir("data")

    for fg_noise in fg_noise_types:
        for bg_noise in bg_noise_types:
            if plat == "win":
                os.mkdir("data\\"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise)
            elif plat == "linux":
                os.mkdir("data/"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise)

    #make dataset

    for i in tqdm(range(examples_per_class)):
        for fg_noise in fg_noise_types:
            for bg_noise in bg_noise_types:
                image_mask,image = make_image_n_mask(x,y,background_noise,fg_noise,bg_noise,shapes[-1],max_shapes=max_num_of_shapes)

                im = Image.fromarray(image_mask*255)
                im1 = Image.fromarray(image)
                if plat == "win":
                    im.save("data\\"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise+"\\"+str(i)+"_mask.png")
                    im1.save("data\\"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise+"\\"+str(i)+"_img.png")
                elif plat == "linux":
                    im.save("data/"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise+"/"+str(i)+"_mask.png")
                    im1.save("data/"+"bg_noise_"+bg_noise+"_VS_fg_noise_"+fg_noise+"/"+str(i)+"_img.png")
