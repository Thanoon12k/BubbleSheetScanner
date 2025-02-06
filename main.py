import cv2
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image
import fitz  
import tkinter as tk
from tkinter import filedialog
from userinterface import UserInterface
from imagemanager import ImageManager
from blocksmanager import BlocksManager

class BubbleSheetScanner:
    def __init__(self,image=None,adaptive_image=None):
        self.bubbles=[]
        self.bubble_radius = 0
        self.average_x_space=0
        self.average_y_space=0
        self.num_bubbles = 0
        self.num_bubbles_blocks=[]
        self.choosen_bubbles=[]
        self.image=image
        self.adaptive_image=adaptive_image
    
    def get_bubble_radius(self):
        bbls=self.bubbles
        rad=0
        for bubble in bbls:
            rad += cv2.boundingRect(bubble[0])[2]
        self.bubble_radius = rad/len(bbls)
        return self.bubble_radius
    
    def get_average_x_and_y_space(self):
        cnts=self.bubbles
        x_space=0 
        y_space=0
        cnts=sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
        for i in range(len(cnts)-1):
            
            x_space += abs(cv2.boundingRect(cnts[i+1])[0]-cv2.boundingRect(cnts[i])[0])


        
        
            y_space += abs(cv2.boundingRect(cnts[i+1])[1]-cv2.boundingRect(cnts[i])[1])
        
        self.average_y_space = y_space/len(cnts)
        self.average_x_space = x_space/len(cnts)

        return self.average_x_space, self.average_y_space
        
    

    def get_bubbles(self ):
        ad_image=self.adaptive_image
        cnts, hierarchy = cv2.findContours(ad_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                self.bubbles.append(c)
        return self.bubbles

    def filter_bubbles(self):
        cnts=self.bubbles
        filtered_bubbles=[]
        shapes={}
        for bubble in cnts:
            approx = cv2.approxPolyDP(bubble, 0.01 * cv2.arcLength(bubble, True), True) 
            # approx = cv2.approxPolyDP(bubble, 0.04 * cv2.arcLength(bubble, True), True)
            if len(approx) == 0:
                shapes["0_app"] = shapes.get("0_app", []) 
                shapes["0_app"].append(bubble)
            elif len(approx) == 1:
                shapes["1_app"] = shapes.get("1_app", []) 
                shapes["1_app"].append(bubble)
            elif len(approx) == 2:
                shapes["2_app"] = shapes.get("2_app", []) 
                shapes["2_app"].append(bubble)
            elif len(approx) == 3:
                shapes["3_app"] = shapes.get("3_app", []) 
                shapes["3_app"].append(bubble)
            elif len(approx) == 4:
                shapes["4_app"] = shapes.get("4_app", []) 
                shapes["4_app"].append(bubble)
            elif len(approx) == 5:
                shapes["5_app"] = shapes.get("5_app", []) 
                shapes["5_app"].append(bubble)
            elif len(approx) == 6:
                shapes["6_app"] = shapes.get("6_app", []) 
                shapes["6_app"].append(bubble)
            elif len(approx) == 7:
                shapes["7_app"] = shapes.get("7_app", []) 
                shapes["7_app"].append(bubble)
            elif len(approx) == 8:
                shapes["8_app"] = shapes.get("8_app", [])
                shapes["8_app"].append(bubble)
            elif len(approx) == 9:
                shapes["9_app"] = shapes.get("9_app", [])
                shapes["9_app"].append(bubble)
            elif len(approx) == 10:
                shapes["10_app"] = shapes.get("10_app", [])
                shapes["10_app"].append(bubble)
            
            else:
                key=f"{len(approx)}_app"
                shapes[key] = shapes.get(key, [])
                shapes[key].append(bubble)
       
            
        return shapes
def main():
    im=ImageManager('blank.png')
    img=im.image
    ad_img=im.adaptive_image
    bss=BubbleSheetScanner(image=img, adaptive_image=ad_img)
    bm=BlocksManager()
    ui=UserInterface()

    bubbles=bss.get_bubbles()

    x_space=0 
    y_space=0
    rad=0
    
    img=bss.image.copy()
    img=ui.draw_bubbles(img, bbls,thickness=9)
    img=ui.put_text(img,f'bubbles: {len(bbls)} ',text_size=2)
    ui.display_image(img,scale_percent=40)
    img=bss.image.copy()
    img=ui.draw_bubbles(img, filterd_bubbles,thickness=9)
    img=ui.put_text(img,f'filtered : {len(filterd_bubbles)} ',text_size=2)
    ui.display_image(img,scale_percent=40)
    
    for i in range(0,len(bbls)-1,2) :
        bb1=bubbles[i]
        bb2=bubbles[i+1]
        x1, y1, w1, h1 = cv2.boundingRect(bb1)
        x2, y2, w2, h2 = cv2.boundingRect(bb2)
        rad+=min(h1,h2) *2
        x_space += abs(x2-x1)    
        y_space += abs(y2-y1)
        img=bss.image
        img=ui.draw_bubbles(img, [bb1,bb2],thickness=9)
        img=ui.put_text(img,f'xs({x1},{x2}) ys({y1},{y2}) dims({w1},{w2},{h1},{h2}) ',text_size=2)
        img=ui.put_text(img,f'rad:{min(w1,w2)}   ,x_sp:{x_space} y_sp:{y_space}',y=120,text_size=2)
        # ui.display_image(img,scale_percent=40)
        
    av_y = y_space/len(bbls)
    av_x = x_space/len(bbls)
    av_rad=rad/len(bbls) 
    print(f"rad:{av_rad} avy:{av_y} avx:{av_x}")

        

    bubbles_groubed_on_shapes=bss.filter_bubbles()
    colors=['gr', 're', 'ye', 'bl', 'cy', 'br','gr', 're', 'ye', 'bl', 'cy', 'br','gr', 're', 'ye', 'bl', 'cy', 'br','gr', 're', 'ye', 'bl', 'cy', 'br']
    # ui.display_image(img,scale_percent=33)
    total_bbls=-4
    for key, value in bubbles_groubed_on_shapes.items():
        c=colors.pop(0)
        total_bbls+=len(value)
        print(f'{key} : {len(value)} color : {c}  bbls {total_bbls}')
        img=bss.image
        img=ui.draw_bubbles(img.copy(), value, c,thickness=9)
        img=ui.put_text(img,f'{key} : {len(value)} bbls: {total_bbls}',text_size=2)
        ui.display_image(img,scale_percent=40)

    
    
    ui.draw_bubbles(img, bubbles)

   

    
    bubble_radius=bss.get_bubble_radius()
    average_x_space, average_y_space=bss.get_average_x_and_y_space()
    print('bubble radius:', bubble_radius, 'average x space:', average_x_space, 'average y space:', average_y_space, 'number of bubbles:', len(bubbles))
    print('application finished')

if __name__ == "__main__":
    main()

    