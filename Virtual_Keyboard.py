import cv2
import numpy as np
keyboard = np.zeros((300,500,3),np.uint8) # height, Width

key_set = {0:"Q",1:"W",2:"E",3:"R",4:"T",
           5:"A",6:"S",7:"D",8:"F",9:"G",
           10:"Z",11:"X",12:"C",13:"V",14:"B"}

def letter(letter_index,text,letter_light):
    if letter_index == 0:
        x,y = 0,0
    elif letter_index == 1:
        x,y = 100,0
    elif letter_index == 2:
        x,y = 200,0
    elif letter_index == 3:
        x,y = 300,0
    elif letter_index == 4:
        x,y = 400,0
    elif letter_index == 5:
        x,y = 0,100
    elif letter_index == 6:
        x,y = 100,100
    elif letter_index == 7:
        x,y = 200,100
    elif letter_index == 8:
        x,y = 300,100
    elif letter_index == 9:
        x,y = 400,100
    elif letter_index == 10:
        x,y = 0,200
    elif letter_index == 11:
        x,y = 100,200
    elif letter_index == 12:
        x,y = 200,200
    elif letter_index == 13:
        x,y = 300,200
    elif letter_index == 14:
        x,y = 400,200

    # Keys
    # x = 0
    # y = 0
    width = 100
    height = 100
    th = 3
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255,0,0), th)
    # Text Settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    #text = "A"
    font_scale = 5
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

#cv2.rectangle(keyboard,(200+th,0+th),(200+width-th,0+height-th),(255,0,0),th)

for i,t in key_set.items():
    if i == 5:
        light = True
    else:
        light = False
    letter(i,t,light)


cv2.imshow("Keyboard",keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()