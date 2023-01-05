import cv2
import mediapipe as mp
import time
import mouse
import numpy as np
from math import degrees, acos
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
click=0
font = cv2.FONT_HERSHEY_PLAIN
pTime = 0
cTime = 0
y10=y12=y16=y20=y18=0
y6=y8=y0=y14=0
mecont=0
# Pulgar
thumb_points = [1, 2, 4]
angle=0
 
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
                if id == 8:
                  cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                  x8,y8=cx,cy
                if id == 6:
                  cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                  y6=cy
                if id==10:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    y10=cy
                if id==12:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    y12=cy
                if id==16:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    y16=cy
                if id==14:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    x14,y14=cx,cy
                #meñique
                if id==20:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    y20=cy
                if id==18:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    y18=cy
                
                for index in thumb_points:
                         x = int(handLms.landmark[index].x * width)
                         y = int(handLms.landmark[index].y * height)
                         coordinates_thumb.append([x, y])
                
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # angulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                #print(angle)

        mouse.move(x8,y8)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
    
        key = cv2.waitKey(1)
        if y10>y12:
            if click==0:
                mouse.click('left')
                click=1
        else:
            click=0
        if y16<y14:
            mecont=mecont+1
            if mecont>20:
                print("ADIOS")
                break
        else:
            mecont=0
        if y20<y18:
            mouse.press('left')
        else:
            mouse.release('left')
        if angle < 115:
            mouse.right_click()
        if key ==ord('x'):
            break
    cv2.imshow("Image", img)
    cv2.waitKey(1)
