import numpy as np
import cv2
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
cap.set(10,80)
face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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
while True:
    success,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
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
            cv2.putText(frame,"Blinking",(50,150),cv2.FONT_HERSHEY_COMPLEX,7,(255,0,0))

        # detecting gaze of an eye
        left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
                                    (landmarks.part(37).x,landmarks.part(37).y),
                                    (landmarks.part(38).x,landmarks.part(38).y),
                                    (landmarks.part(39).x,landmarks.part(39).y),
                                    (landmarks.part(40).x,landmarks.part(40).y),
                                    (landmarks.part(41).x,landmarks.part(41).y)],np.int32)
        #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)

        # creating mask for to select the left eye region accurately
        height,width,_ = frame.shape
        mask = np.zeros((height,width),np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask,[left_eye_region],255)
        left_eye = cv2.bitwise_and(gray,gray,mask=mask)


        min_x = np.min(left_eye_region[:,0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:,1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = left_eye[min_y:max_y,min_x:max_x]
        #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(gray_eye,None,fx=5,fy=5)
        cv2.imshow("Eye",eye) # gray_eye
        # threshold for left eye
        _,threshold_eye = cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
        height,width = threshold_eye.shape
        left_side_threshold = threshold_eye[0:height,0:int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0:height, int(width/2):width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        gaze_ratio = left_side_white/right_side_white

        # cv2.putText(frame,str(left_side_white),(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        # cv2.putText(frame, str(right_side_white), (20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio), (20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        cv2.imshow("Threshold_Eye", threshold_eye)
        # in this we need to remove the skin part in eye
        # so for that we create a mask. it means blank image and we place the
        # mask on the frame and we extract the eye then we can find the eye regions with accurately
        # cv2.imshow("Left_gray_eye",left_eye)
        # left_side_threshold = cv2.resize(left_side_threshold,None,fx=8,fy=8)
        # right_side_threshold = cv2.resize(right_side_threshold, None, fx=8, fy=8)
        cv2.imshow("Left",left_side_threshold)
        cv2.imshow("Right",right_side_threshold)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



