import os
import numpy as np
import cv2
import mediapipe as mp
from os import listdir
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
HEIGHT= 200
WIDTH= 200
# Indicate the path to access the directory
data_path="data"
data= os.listdir(data_path)
# Run each file through the path and save it in "sign_language"
# Run each image through the path then save it in IMAGE_FILES
IMAGE_FILES=[]
limite = 1
for folder in data:
  contador=0
  sign_language= os.listdir(f"{data_path}/{folder}")
  for images in sign_language:
    if contador < limite:
      IMAGE_FILES.append(f"{data_path}/{folder}/{images}")
      contador+= 1
print(len(IMAGE_FILES))
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.image.png
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
   
    if not results.multi_hand_landmarks:
      continue
    for hand_landmarks in results.multi_hand_landmarks:
      print (hand_landmarks.landmark[0].x, dir(hand_landmarks.landmark[0]))
      break
      #print('hand_landmarks:', hand_landmarks)
      #print(
          #f'Index finger tip coordinates: (',
          #f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          #f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      #)
      #mp_drawing.draw_landmarks(
          #annotated_image,
          #hand_landmarks,
          #mp_hands.HAND_CONNECTIONS,
          #mp_drawing_styles.get_default_hand_landmarks_style(),
          #mp_drawing_styles.get_default_hand_connections_style())
    #cv2.imwrite(
        #'/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    #if not results.multi_hand_world_landmarks:
    # continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    # mp_drawing.plot_landmarks(
    # hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)