from skimage.filters import gaussian
import cv2
import numpy as np

# Função para extrair as faces da imagem
def detectFaceOpenCVDnn(net, frame, conf_threshold=0.7):
    faces = []

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.8, (800, 800), [104, 117, 123], True, False, )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            top = x1
            right = y1
            bottom = x2 - x1
            left = y2 - y1

            # Extraímos as faces detectadas
            face = frame[right:right + left, top:top + bottom]
            faces.append(face)

    return faces

# --------------------------------------------------------------------------------
# Detecção de pessoas com YOLO em  imagens
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


def detectFaceOpenCV(net, frames, conf_threshold=0.7):
    faces = []

    for i in range(len(frames)):

        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800), [104, 117, 123], False, False, )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                top = x1
                right = y1
                bottom = x2 - x1
                left = y2 - y1

                # Extraímos as faces detectadas
                face = frame[right:right + left, top:top + bottom]
                faces.append(face)

    return faces

