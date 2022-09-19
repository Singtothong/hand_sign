import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
tipIds= [4,8,12,16,20]
mp_draw= mp.solutions.drawing_utils
from PIL import Image

# For static images:
# IMAGE_FILES = []
image = cv2.imread('360_F_61499604_hkfPSZ4ZYs47Yp8H780DEb3I3cvWjdmH.jpg', cv2.IMREAD_UNCHANGED)
# IMAGE_FILES.append(image)
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
#   for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    # image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    # if not results.multi_hand_landmarks:
    #   continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    lmList=[]
    lmList2forModel=[]
    for hand_landmarks in results.multi_hand_landmarks:
        
    #   print('hand_landmarks:', hand_landmarks)
    #   print(
    #       f'Index finger tip coordinates: (',
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
    #   )
        # overlay = cv2.imread('360_F_61499604_hkfPSZ4ZYs47Yp8H780DEb3I3cvWjdmH.jpg')
       
        # x_offset = 30
        # y_offset = 170
        # x_end = x_offset + overlay.shape[1]
        # y_end = y_offset + overlay.shape[0]
        # annotated_image[y_offset:y_end,x_offset:x_end] = overlay
        # cv2_imshow(large_img)
        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
      
        myHands=results.multi_hand_landmarks[0]
        # print(myHands)
        
        # h,w,c=image.shape
        # cx,cy=int(myHands.landmark[0].x*w), int(myHands.landmark[0].y*h)
        for id,lm in enumerate(myHands.landmark):
            h,w,c=image.shape
            cx,cy=int(lm.x*w), int(lm.y*h)
            lmList.append([id,cx,cy])
            lmList2forModel.append([cx,cy])
        lmList.append([cx,cy])
        lmList2forModel.append([cx,cy])
        fingers=[]
        print(lmList)
        print(lmList[0])
        print(lmList[0][0])
        print(tipIds[0])
        print(lmList[1][0])
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)

        else:
            fingers.append(0)


        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-1][2]:
                fingers.append(1)


            else:
                fingers.append(0)

        total= fingers.count(1)
        print('**********')
        print(total)
        if total==4:
            print('brake')
    # cv2.imwrite(
    #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    cv2.imshow('frame',annotated_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
        
    
    # Draw hand world landmarks.
    # if not results.multi_hand_world_landmarks:
    #   continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    #   mp_drawing.plot_landmarks(
    #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    #   cv2.imshow('frame',annotated_image)