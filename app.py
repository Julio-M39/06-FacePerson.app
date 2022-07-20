# Modified by Júlio Monteiro 2022
# Interface de usuário para o reconhecimento facial

# Pacotes utilizados
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import skimage
from skimage.filters import gaussian
from skimage import color, data
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import time
from skimage.feature import Cascade
from PIL import Image
from functions import detectFaceOpenCVDnn, detectFaceOpenCV, yolo_person, blob_imagem, deteccoes, extrair_person

# ------ Detecção de rostos -------------
modelFile = "cfg/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "cfg/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# ------ Detecção de pessoas ------------


st.title('FacePerson (Detecção de rostos e pessoas em imagens e vídeos)')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('FacePerson (Detecção de rostos e pessoas em imagens e vídeos)')

app_mode = st.sidebar.selectbox('Escolha o modo a ser utilizado',
                                ['Sobre o aplicativo', 'Imagens', 'Vídeos']
                                )

# Informações sobre o aplicativo

if app_mode == 'Sobre o aplicativo':
    st.markdown(
        'A detecção de rostos e de pessoas é o primeiro passo que antecede o reconhecimento facial ou o reconhecimento de pessoas em um determinado perímetro – a detecção vai apontar, a partir de uma determinada imagem ou vídeo, se ali existe um rosto ou uma pessoa.')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image('capa.png')

    #st.markdown('''
          # About Me \n 
            #Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n

            #If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
            #www.augmentedstartups.info/visionstore \n

            #Also check us out on Social Media
            #- [YouTube](https://augmentedstartups.info/YouTube)
            #- [LinkedIn](https://augmentedstartups.info/LinkedIn)
            #- [Facebook](https://augmentedstartups.info/Facebook)
            #- [Discord](https://augmentedstartups.info/Discord)

            #If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)

            #''')

# Utilização do modo imagens
elif app_mode == 'Imagens':

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mode_image = st.sidebar.selectbox('Modos de detecção',
                                    ['Escolher modos','Detecção de rostos', 'Detecção de pessoas']
                                    )

    img_file_buffer = st.sidebar.file_uploader("Carregue uma imagem", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Imagem original.')
        st.sidebar.image(image)

    # Detecção de rostos
    if mode_image == 'Escolher modos':
        st.subheader('Escolha o modo de detecção e faça o upload de uma imagem')


    # Detecção de rostos
    if mode_image == 'Detecção de rostos':

        if img_file_buffer is None:
            st.subheader('Faça o upload de uma imagem')
        else:

            imagens = detectFaceOpenCVDnn(net, image)

            faces = []

            for i in range(len(imagens)):
                img = imagens[i].shape
                if img[0] > 5 and img[1] > 5:
                    faces.append(imagens[i])

            if len(faces) > 0:
                st.subheader('Rostos encontrados')
                st.image(faces, use_column_width=100, width=100)
            else:
                st.subheader('Nenhum rosto encontrado')

    # Detecção de pessoas
    elif mode_image == 'Detecção de pessoas':

        if img_file_buffer is None:
            st.subheader('Faça o upload de uma imagem')
        else:
            net, ln, LABELS = yolo_person()
            threshold = 0.7
            threshold_NMS = 0.3
            classes = ['person']

            imagem = image
            (H, W) = imagem.shape[:2]

            imagem_cp = imagem.copy()
            net, imagem, layer_outputs = blob_imagem(net, imagem, ln)

            caixas = []
            confiancas = []
            IDclasses = []

            for output in layer_outputs:
                for detection in output:
                    caixas, confiancas, IDclasses = deteccoes(detection, threshold, caixas, confiancas, IDclasses, W, H)

            objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, threshold_NMS)

            pessoas = extrair_person(objs, caixas, LABELS, IDclasses, classes, imagem_cp)

            persons = []

            for i in range(len(pessoas)):
                img = pessoas[i].shape
                if img[0] > 50 and img[1] > 50:
                    persons.append(pessoas[i])

            if len(persons) > 0:
                st.subheader('Pessoas encontradas')
                st.image(persons, use_column_width=100, width=100)
            else:
                st.subheader('Nenhuma pessoa encontradas')

elif app_mode =='Vídeos':

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    mode_video = st.sidebar.selectbox('Modos de detecção',
                                      ['Escolher modos', 'Detecção de rostos', 'Detecção de pessoas']
                                      )

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Carregue um vídeo", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer is not None:
        tfflie.write(video_file_buffer.read())
        vidcap = cv2.VideoCapture(tfflie.name)
        st.sidebar.text('Vídeo original')
        st.sidebar.video(tfflie.name)


    # Detecção de rostos
    if mode_video == 'Escolher modos':
        st.subheader('Escolha o modo de detecção e faça o upload de um vídeo')

    # Detecção de rostos
    if mode_video == 'Detecção de rostos':
        if video_file_buffer is None:
            st.subheader('Faça o upload de um vídeo')
        else:
            frames = []
            def getFrame(sec):
                vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                hasFrames, image = vidcap.read()
                if hasFrames:
                    frames.append(image)
                return hasFrames

            sec = 0
            frameRate = 2
            success = getFrame(sec)

            while success:
                sec = sec + frameRate
                sec = round(sec, 2)
                success = getFrame(sec)

            imags = detectFaceOpenCV(net, frames)

            faces = []

            for i in range(len(imags)):
                img = imags[i].shape
                if img[0] > 15 and img[1] > 15:
                    faces.append(imags[i])

            if len(faces) > 0:
                st.subheader('Rostos encontrados')
                st.image(faces, use_column_width=100, width=100)
            else:
                st.subheader('Nenhum rosto encontrado')

    # Detecção de pessoas
    if mode_video == 'Detecção de pessoas':
        if video_file_buffer is None:
            st.subheader('Faça o upload de um vídeo')
        else:
            frames = []
            def getFrame(sec):
                vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                hasFrames, image = vidcap.read()
                if hasFrames:
                    frames.append(image)
                return hasFrames

            sec = 0
            frameRate = 2
            success = getFrame(sec)

            while success:
                sec = sec + frameRate
                sec = round(sec, 2)
                success = getFrame(sec)

            net, ln, LABELS = yolo_person()
            threshold = 0.5
            threshold_NMS = 0.3
            classes = ['person']

            pessoasimg = []

            for i in range(len(frames)):
                imagem = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                (H, W) = imagem.shape[:2]

                imagem_cp = imagem.copy()
                net, imagem, layer_outputs = blob_imagem(net, imagem, ln)

                caixas = []
                confiancas = []
                IDclasses = []

                for output in layer_outputs:
                    for detection in output:
                        caixas, confiancas, IDclasses = deteccoes(detection, threshold, caixas, confiancas, IDclasses,
                                                                  W, H)

                objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, threshold_NMS)

                pessoas = extrair_person(objs, caixas, LABELS, IDclasses, classes, imagem_cp)

                for i in range(len(pessoas)):
                    pessoasimg.append(pessoas[i])

            persons = []

            for i in range(len(pessoasimg)):
                img = pessoasimg[i].shape
                if img[0] > 50 and img[1] > 50:
                    persons.append(pessoasimg[i])

            if len(persons) > 0:
                st.subheader('Pessoas encontradas')
                st.image(persons, use_column_width=150, width=150)
            else:
                st.subheader('Nenhuma pessoa encontradas')














