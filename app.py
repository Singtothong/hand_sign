# from bokeh.themes import theme
# from numpy.core.records import record
# import pandas as pd
# import numpy as np
import cv2
import mediapipe as mp
import time
# from PIL import Image
import tempfile
# from bokeh.models.widgets import Div
import streamlit as st
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
st.title('Sign Language Recognition')
# ----------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ----------------------------------------------------------------------

t = st.empty()
def draw(str):
    t.markdown(f'<p style="font-family:Arial Black;color:#FF0686;font-size:28px;;">{str}</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
width: 350px
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
width: 350px
margin-left: -350px
</style>
    """,unsafe_allow_html=True,)
# ----------------------------------------------------------------------


@st.cache ()
# ----------------------------------------------------------------------

def image_resize(image, width=None, height=None, inter =cv2.INTER_AREA):
    
    dim = None
    (h ,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r= width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    #resize the image
    resized =cv2.resize(image, dim ,interpolation=inter)
    return resized


# ----------------------------------------------------------------------

app_mode= st.sidebar.selectbox('Choose the App Mode',
                               ['Run On Video'])

symptom = ''

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    #image = cv2.imread(file), 1
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

if app_mode == 'Run On Video':
    
    st.set_option('deprecation.showfileUploaderEncoding',False)
    
    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """,unsafe_allow_html=True,)

    st.sidebar.markdown('---')
    
    st.subheader("Input Video")    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    #tffile = tempfile.NamedTemporaryFile()

    str_video = str(video_file_buffer)
    x = str_video.split()
    if len(x) > 1:
        print(x[1][6:-6])
        x = x[1][6:-6]
        if x[-1] == '0':
            symptom = 'สายตาเอียง'
        elif x[-1] == '1':
            symptom = 'มึนศีรษะ'
        elif x[-1] == '2':
            symptom = 'เป็นหวัด'
        elif x[-1] == '3':
            symptom = 'ความดันโลหิตสูง'
        elif x[-1] == '4':
            symptom = 'แสบจมูก'
        elif x[-1] == '5':
            symptom = 'เมื่อย'
        

    #We get our input video here
    if video_file_buffer:
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(video_file_buffer.read())
            vid = cv2.VideoCapture(temp.name)
            fps_input = int(vid.get(cv2.CAP_PROP_FPS))
            stframe = st.empty()
    #Recording Part
            #codec = cv2.VideoWriter_fourcc('V', 'P', '0','9')
            
            fps = 0
            i = 0
            prevTime=0
            currTime=0 
            #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                original_title = '<p style="text-align: center; font-size: 20px;"><strong>Frame Rate</strong></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                kpi1_text = st.markdown ("0")
            with kpi2:
                original_title = '<p style="text-align: center; font-size: 20px;"><strong>Detected Hands</strong></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                kpi2_text = st.markdown ("0")
            with kpi3:
                original_title = '<p style="text-align: center; font-size: 20px;"><strong>Symptom</strong></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                kpi3_text = st.markdown("")
                
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                while vid.isOpened():
                    success, image = vid.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        break

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #print(results)
                    hand_count=0
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            hand_count += 1
                            mp_drawing.draw_landmarks(
                            image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                    currTime = time.time()
                    fps = 1/ (currTime - prevTime)
                    prevTime = currTime
                    #image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{int(fps)}</h1>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
                    # kpi3_text.write(f"<h1 style='text-align: center; color:red; '>{symptom}</h1>", unsafe_allow_html=True)
                    
                    image = cv2.resize(image, (0,0), fx = 0.8, fy =0.8)
                    image = image_resize(image = image, width = 320,height=360)
                    stframe.image(image, channels = 'BGR', use_column_width=False) 
                    
                kpi3_text.write(f"<h1 style='text-align: center; color:red; '>{symptom}</h1>", unsafe_allow_html=True)
            vid.release()
