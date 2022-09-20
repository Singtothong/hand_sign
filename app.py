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

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'test3.mp4'
prevTime=0
currTime=0 
tipIds= [4,8,12,16,20]
st.title('Sign Language Recognition')

# ----------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_draw= mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_draw= mp.solutions.drawing_utils
mp_hand= mp.solutions.hands
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

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


if app_mode == 'Run On Video':
    
    st.set_option('deprecation.showfileUploaderEncoding',False)
    # use_webcam = st.sidebar.button('Use Webcam')
    use_webcam = 0
    
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

    
    # max_hands= st.sidebar.number_input('Maximum Number of Hand',value=1,min_value=1,max_value=4)
    max_hands = 2
    # detection_confidence= st.sidebar.slider('Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    detection_confidence = 0.5
    # tracking_confidence= st.sidebar.slider('Tracking Confidence Confidence',min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence = 0.5
    
    st.sidebar.markdown('---')
    
    # st.subheader("Input Video")    
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    #We get our input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    
    #Recording Part
    codec = cv2.VideoWriter_fourcc('V', 'P', '0','9')
    out= cv2.VideoWriter('output.mp4',codec,fps_input,(width,height))
    
    # st.sidebar.text('Input Video')
    # st.sidebar.video(tffile.name)
     
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
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
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Video Width</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi3_text = st.markdown("0")
        
    with mp_hand.Hands(max_num_hands=max_hands,min_detection_confidence=detection_confidence,
                       min_tracking_confidence=tracking_confidence) as hands:
    
        
        while vid.isOpened():
            
            i +=1
            ret, image = vid.read()
            if not ret:
                continue
        
          
            image.flags.writeable=False
            results= hands.process(image)
            image.flags.writeable=True
            image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            lmList=[]
            lmList2forModel=[]
            hand_count=0
            if results.multi_hand_landmarks:
                
                for hand_landmark in results.multi_hand_landmarks:
                    hand_count += 1
                    myHands=results.multi_hand_landmarks[0]
                    for id,lm in enumerate(myHands.landmark):
                        h,w,c=image.shape
                        cx,cy=int(lm.x*w), int(lm.y*h)
                        lmList.append([id,cx,cy])
                        lmList2forModel.append([cx,cy])
                    
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
                    if total==5:
                        sh= "Acclerate"
                        #draw(sh)
                    if total==2 or total==3:
                        sh= "Left"
                        #draw(sh)
                    if total==4:
                        sh= "Right"
                        #draw(sh)
                    if total==0:
                        sh= "Brake"
                        #draw(sh)
                    
                    mp_draw.draw_landmarks(image,hand_landmark,mp_hand.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
                    
                #FPS Counter Logic
            currTime = time.time()
            fps = 1/ (currTime - prevTime)
            prevTime = currTime
            fingers=[]
            
            # if record:
            #     out.write(image)
            image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
            
            kpi3_text.write(f"<h1 style='text-align: center; color:red; '>{width}</h1>", unsafe_allow_html=True)
            
            image = cv2.resize(image, (0,0), fx = 0.8, fy =0.8)
            image = image_resize(image = image, width = 320,height=360)
            stframe.image(image, channels = 'BGR', use_column_width=False)
    st.subheader('Output Image')
    st.text('Video Processed')
    output_video = open('output1.mp4','rb')
    out_bytes= output_video.read()
    st.video(out_bytes)
    

    # st.video(video_bytes) 
    vid.release()
    out.release()
 