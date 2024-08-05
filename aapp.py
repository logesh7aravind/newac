import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image, ImageDraw

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import joblib

#training the classifier using X_Train and y_train 

# OJO sólo activar siguiente linea cuando se desea hacer un entrenamiento
#clf = SVC(kernel = 'linear').fit(X_train,y_train) 

try:
    clf = joblib.load("svm/mediaPipe.pkl")
    print("using trained model")
except:
    print("building new model")
    #clf = SVC(kernel = 'linear').fit(X_train,y_train)
    #clf.fit(X_train, y_train)
    joblib.dump(clf,"svm/mediaPipe.pk")
    
def emotionImage(emotion):
	# Emojis
	if emotion == 'Disgusto': image = cv.imread('Emojis/disgusto.jpeg')
	if emotion == 'Enojo': image = cv.imread('Emojis/enojo.jpeg')
	if emotion == 'Felicidad': image = cv.imread('Emojis/felicidad.jpeg')
	if emotion == 'Miedo': image = cv.imread('Emojis/miedo.jpeg')
	if emotion == 'Somnoliento': image = cv.imread('Emojis/somnoliento.jpeg')        
	if emotion == 'Sorpresa': image = cv.imread('Emojis/sorpresa.jpeg')
	if emotion == 'Tristeza': image = cv.imread('Emojis/tristeza.jpeg')
	return image

dataPath = 'Dir_drow' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

# Basic App Scaffolding
st.title('Facial Expressions Recognition usando VSM, Mediapipe y Streamlit')

## Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## Create Sidebar
st.sidebar.title('Reconocimiento FER')
st.sidebar.subheader('Parámetros')

## Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'Modo',
    ['About','Image','Video']
)

# Resize Images to fit Container
@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized


# About Page

if app_mode == 'About':
    st.markdown('''
                ## Face Mesh \n
                En esta apliación usamos **MediaPipe** para crear a partir de Face Mesh,
                un sistema de reconocimiento de Expresiones Faciales. **StreamLit** 
                permite crear La interfaz de Usuario Gráfica WEB (GUI) \n
                
                - [Repositorio Github](https://github.com/OTapias/Streamlit-FER) \n
    ''')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Image Page

elif app_mode == 'Image':
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Faces**")
    kpil_text = st.markdown('0')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    ## Output
    st.markdown('## Output')
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count=0

    ## Dashboard
    
#    IMAGE_FILES = []
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
#        for idx, file in enumerate(IMAGE_FILES):
#            image = cv2.imread(file)
        
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Draw face detections of each face.
#        if not results.detections:
#            continue
        annotated_image = image.copy()
        for detection in results.detections:
            face_count += 1
            #print('Nose tip:')
            #print(mp_face_detection.get_key_point(
            #    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
            #cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)  
            kpil_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(annotated_image, use_column_width=True)
    
    
 

# Video Page

elif app_mode == 'Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Usar Webcam')
    record = st.sidebar.checkbox("Grabar Video")

    if record:
        st.checkbox('Grabando', True)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    max_faces = st.sidebar.number_input('Número Máximo de Rostros', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Mín. Factor de Detección', min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence = st.sidebar.slider('Mín. Factor de Seguimiento', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    ## Get Video
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Cargar un Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv.VideoCapture(0)
        else:
            video = cv.VideoCapture(DEMO_VIDEO)
            temp_file.name = DEMO_VIDEO

    else:
        temp_file.write(video_file_buffer.read())
        video = cv.VideoCapture(temp_file.name)

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv.CAP_PROP_FPS))

    ## Recording
    codec = cv.VideoWriter_fourcc('a','v','c','1')
    out = cv.VideoWriter('output1.mp4', codec, fps_input, (width,height))

    st.sidebar.text('Video de Entrada')
    st.sidebar.video(temp_file.name)

    fps = 0
    i = 0

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    kpil, kpil2, kpil3 = st.columns(3)

    with kpil:
        st.markdown('**Tasa de Muestreo**')
        kpil_text = st.markdown('0')

    with kpil2:
        st.markdown('**Rostros Detectados**')
        kpil2_text = st.markdown('0')

    with kpil3:
        st.markdown('**Resolución de Imagen**')
        kpil3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)


    ## Face Mesh
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence

    ) as face_mesh:

            prevTime = 0

            while video.isOpened():
                i +=1
                ret, frame = video.read()
                if not ret:
                    continue

                x_data = []
                y_data = []
                r_data = []
                
                image_2 = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                height, width, _ = image_2.shape
                image_rgb = cv.cvtColor(image_2, cv.COLOR_BGR2RGB)
                
                
                results = face_mesh.process(frame)
                frame.flags.writeable = True
                
                
                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        for dato in face_landmarks.landmark:
                            x = int(dato.x * width)
                            y = int(dato.y * height)
                            #cv2.circle(image, (x,y), 2, (255,0,0), 2)
                            x_data.append(dato.x)
                            y_data.append(dato.y)
                            #facesData.append((x,y))

                x_min = min(x_data)
                x_max = max(x_data)
                y_min = min(y_data)
                y_max = max(y_data)

                x_data = [(float(i) - x_min)/(x_max - x_min) for i in x_data]
                y_data = [(float(i) - y_min)/(y_max - y_min) for i in y_data] 

                r_data = np.sqrt(np.square(x_data) + np.square(y_data))
                
                df = pd.DataFrame(r_data)
                df = df.T
                #print(df)
                res = clf.predict(df)
                #print(res[0])
                image_2 = emotionImage(imagePaths[int(res[0])]) 
                
                with mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5) as face_detection:
                        results = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                        # Draw face detections of each face.
                        if not results.detections:
                            continue
                        annotated_image = frame.copy()
                        for detection in results.detections:
                            #mp_drawing.draw_detection(annotated_image, detection)
                            w_coor = int(detection.location_data.relative_bounding_box.width * width)
                            h_coor = int(detection.location_data.relative_bounding_box.height * height)                
                            x_coor = int(detection.location_data.relative_bounding_box.xmin * width)
                            #x_coor = int(detection.location_data.relative_keypoints[0].x * width) ubicación de puntos que alumbran
                            y_coor = int(detection.location_data.relative_bounding_box.ymin * height)
                            #y_coor = int(detection.location_data.relative_keypoints[0].y * height) ubicación de puntos que alumbran
                            cv.rectangle(annotated_image, (x_coor,y_coor),(x_coor+w_coor,y_coor+h_coor),(255,255,0),2)

                cv.putText(annotated_image,'{}'.format(imagePaths[int(res[0])]),(x_coor,y_coor-5),1,1.7,(255,255,0),1,cv.LINE_AA)
                
                im = emotionImage(imagePaths[int(res[0])])
                
#                im_h_resize = hconcat_resize_min([annotated_image, im])
                
#                cv2.imshow("Frame", im_h_resize)


                face_count = 0
#                if results.multi_face_landmarks:

                    #Face Landmark Drawing
#                    for face_landmarks in results.multi_face_landmarks:
#                        face_count += 1

#                       mp.solutions.drawing_utils.draw_landmarks(
#                            image=frame,
#                            landmark_list=face_landmarks,
#                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#                            landmark_drawing_spec=drawing_spec,
#                            connection_drawing_spec=drawing_spec
#                        )

                # FPS Counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                # Dashboard
                kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                                 unsafe_allow_html=True)


                frame = cv.resize(annotated_image,(0,0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame,channels='BGR', use_column_width=True)

#                frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
#                frame = image_resize(image=frame, width=640)
#                stframe.image(frame,channels='BGR', use_column_width=True)
