import cv2
import mediapipe as mp
import mouse
import numpy as np
from math import degrees, acos
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
click=0
mecont=0
# Pulgar
thumb_points = [1, 2, 4]
fingers = [6,8,10,12,14,16,18,20]
angle=0
coord_fingers = []
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  #doy vuelta la imagen para que este alineada a la persona
    height, width, _ = img.shape
    img = cv2.resize(img, (1920, 1080))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        coordinates_thumb = []
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                for index in fingers:
                    if id == index:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        coord_fingers.append([cx, cy])
                for pulg in thumb_points:
                    x = int(handLms.landmark[pulg].x * width)
                    y = int(handLms.landmark[pulg].y * height)
                    coordinates_thumb.append([x, y])
                
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # angulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
        mouse.move(coord_fingers[1][0],coord_fingers[1][1])
    
        key = cv2.waitKey(1)
        fingers = [6,8,10,12,14,16,18,20]
        if coord_fingers[2][1]>coord_fingers[3][1]:
            if click==0:
                mouse.click('left')
                click=1
        else:
            click=0
        if coord_fingers[5][1]<coord_fingers[4][1]:
            mecont=mecont+1
            if mecont>20:
                print("Stop")
                break
        else:
            mecont=0
        if coord_fingers[7][1]<coord_fingers[6][1]:
            mouse.press('left')
        else:
            mouse.release('left')
        if angle < 117:
            mouse.right_click()
        if key ==ord('x'):
            break
    coord_fingers = []
    cv2.imshow("Image", img)
