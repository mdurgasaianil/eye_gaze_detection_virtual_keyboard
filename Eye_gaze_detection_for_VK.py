import numpy as np
import cv2
import dlib
import random
from math import hypot
cap = cv2.VideoCapture(0)
cap.set(10,100)
board = np.zeros((400,500),np.uint8)
board[:] = 255
face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# keyboard settings
keyboard = np.zeros((300,500,3),np.uint8) # height, Width
key_set_1= {0:"Q",1:"W",2:"E",3:"R",4:"T",
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


def midpoint(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)

def get_blink_ratio(eye_points,facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length/ver_line_length
    return ratio

def get_gaze_ratio(eye_points,facial_landmarks,):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)

    # creating mask for to select the left eye region accurately
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y:max_y, min_x:max_x]
    # gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    # eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    # cv2.imshow("Eye", eye)  # gray_eye
    # threshold for left eye
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white != 0 and right_side_white != 0:
        gaze_ratio = left_side_white / right_side_white
    else:
        gaze_ratio = 1
    return gaze_ratio,threshold_eye,gray_eye

font = cv2.FONT_HERSHEY_COMPLEX
# counters
frames = 0
letter_index = 0
blinking_frames = 0
text = ""
while True:
    success,frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5,fy=0.5) # resizing the frame
    keyboard[:] = (0,0,0) # color black
    frames += 1 # frames = frames + 1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if letter_index<=14:
        active_letter = key_set_1[letter_index]
    else:
        active_letter = key_set_1[0]
    faces = face_detector(gray)
    if len(faces) != 0:
        for face in faces:
            x,y = face.left(),face.top()
            w,h = face.right(),face.bottom()
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),3)
            landmarks = landmarks_predictor(gray,face) # predicting 68 landmarks from the current frame
            # detecting blinking
            left_eye_ratio = get_blink_ratio([36,37,38,39,40,41],landmarks)
            right_eye_ratio = get_blink_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
            #print(blinking_ratio)
            if blinking_ratio > 5:
                cv2.putText(frame,"Blinking",(20,150),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
                blinking_frames += 1
                frames -= 1
                if blinking_frames == 5:
                    text += active_letter
                    #print(text)
            else:
                blinking_frames = 0
            # detecting gaze of an eye
            gaze_ratio_left_eye,left_threshold_eye,left_gray_eye = get_gaze_ratio([36,37,38,39,40,41],landmarks)
            gaze_ratio_right_eye,right_threshold_eye,right_gray_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47],landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            if gaze_ratio < 0.5:
                cv2.putText(frame, "Right", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            elif 0.5 <= gaze_ratio < 2 :
                cv2.putText(frame, "Center", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            elif gaze_ratio >=2:
                cv2.putText(frame, "Left", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            left_threshold_eye = cv2.resize(left_threshold_eye,None,fx=5,fy=5)
            right_threshold_eye = cv2.resize(right_threshold_eye, None, fx=5, fy=5)
            cv2.imshow("left_threshold_eye", left_threshold_eye)
            cv2.imshow("right_threshold_eye", right_threshold_eye)
        if frames == 30:
            if letter_index <= 14:
                letter_index += 1
            else:
                letter_index = 0
            frames = 0

        for i, t in key_set_1.items():
            if i == letter_index:
                light = True
            else:
                light = False
            letter(i, t, light)
        cv2.putText(board,text,(10,50),font,2,0,3)
    cv2.imshow("Frame",frame)
    cv2.imshow("Virtual Keyboard",keyboard)
    cv2.imshow("board",board)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



