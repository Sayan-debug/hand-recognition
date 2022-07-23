import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLms.landmark):
                # print(id, landmark)
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(id, cx, cy)
                # For dectecting a particular point in hand
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (225, 225, 225), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (252, 3, 225), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1) 


