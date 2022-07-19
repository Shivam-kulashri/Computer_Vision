import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # default parameters (static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                # print(id,lm)
                h , w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)    # converting length breadth and height into pixels from numeric value
                print(id, cx, cy)
                if id == 4 | 0:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(img, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(1)