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
from functions import extrai_face, yolo_person, blob_imagem, deteccoes, funcoes_imagem_v2, extrair_person

# ------ Detecção de rostos -------------
# Carrega o arquivo de treino para o detector facial
arquivo_treino = data.lbp_frontal_face_cascade_filename()
# Criamos o detector
detector = Cascade(arquivo_treino)

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
    st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')

    st.markdown('''
          # About Me \n 
            Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n

            If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
            www.augmentedstartups.info/visionstore \n

            Also check us out on Social Media
            - [YouTube](https://augmentedstartups.info/YouTube)
            - [LinkedIn](https://augmentedstartups.info/LinkedIn)
            - [Facebook](https://augmentedstartups.info/Facebook)
            - [Discord](https://augmentedstartups.info/Discord)

            If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)

            ''')

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

            # Detecção de faces na imagem
            faces_detectadas_1 = detector.detect_multi_scale(img=image,
                                                             scale_factor=2.0,
                                                             step_ratio=1,
                                                             min_size=(5, 5),
                                                             max_size=(300, 300))

            face = extrai_face(faces_detectadas_1, image)

            if len(face) > 0:
                st.subheader('Rostos encontrados')
                st.image(face, use_column_width=150, width=150)
            else:
                st.subheader('Nenhum rosto encontrado')

    # Detecção de pessoas
    elif mode_image == 'Detecção de pessoas':

        if img_file_buffer is None:
            st.subheader('Faça o upload de uma imagem')
        else:
            net, ln, LABELS = yolo_person()
            threshold = 0.5
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

            if len(pessoas) > 0:
                st.subheader('Pessoas encontradas')
                st.image(pessoas, use_column_width=150, width=150)
            else:
                st.subheader('Nenhuma pessoa encontradas')












