from skimage.filters import gaussian
import cv2
import numpy as np

# Função para extrair as faces da imagem
def extrai_face(faces_detectadas_1, image):
    face_face = []
    for i in range(len(faces_detectadas_1)):
        d = faces_detectadas_1[i]

        # Aqui, x e y representam os pontos das coordenadas da área da face detectada
        x, y = d['r'], d['c']

        # Largura e altura da área da face detectada
        width, height = d['r'] + d['width'], d['c'] + d['height']

        # Extraímos as faces detectadas
        face = image[x:width, y:height]

        face_face.append(face)

    return face_face

# --------------------------------------------------------------------------------
# Detecção de pessoas com YOLO

def yolo_person():
    labels_path = 'cfg/coco.names'
    LABELS = open(labels_path).read().strip().split('\n')
    weights_path = 'cfg/yolov4.weights'
    config_path = 'cfg/yolov4.cfg'
    net = cv2.dnn.readNet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln, LABELS

def blob_imagem(net, imagem, ln):

    blob = cv2.dnn.blobFromImage(imagem, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    return net, imagem, layerOutputs

def deteccoes(detection, threshold, caixas, confiancas, IDclasses, W, H):
    scores = detection[5:]
    classeID = np.argmax(scores)
    confianca = scores[classeID]

    if confianca > threshold:
        caixa = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = caixa.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        caixas.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        IDclasses.append(classeID)

    return caixas, confiancas, IDclasses

def funcoes_imagem_v2(i, caixas):
    (x,y) = (caixas[i][0], caixas[i][1]) # Coordenada (x,y) onde inicia a caixa da detecção
    (w,h) = (caixas[i][2], caixas[i][3]) # Largura e altura em pixels da caixa de detecção

    return x,y,w,h


def extrair_person(objs, caixas, LABELS, IDclasses, classes, imagem_cp):
    persons = []

    if len(objs) > 0:
        for i in objs.flatten():
            if LABELS[IDclasses[i]] in classes:
                x, y, w, h = funcoes_imagem_v2(i, caixas)
                objeto = imagem_cp[y:y + h, x:x + w]
                persons.append(objeto)
    return persons



