import xml.etree.ElementTree as ET
import os
import math
import cv2
import face_detection


'''
    Cria arquivo .xml com a descrição da imagem e as anotações de cada landmark

    Estrutura base:

    <?xml version='1.0' encoding='utf-8'?>
    <dataset>
    <name />
    <comment />
    <images>
        <image file="image_064_1.jpg" width="1024" height="657">
            <box top="152" left="315" width="288" height="413">
                <part name="0" x="295" y="339" />
            </box>
        </image>
    </images>
    </dataset>
'''

def str_trunc(x):
    return str(math.trunc(float(x)))

# Inicializa detector de faces
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

# Cria estrutura do xml
root = ET.Element('dataset')
name = ET.SubElement(root, 'name')
comment = ET.SubElement(root, 'comment')
images = ET.SubElement(root, 'images')

# Percorre o diretório de imagens do dataset
datasetPath = './data/ibug'
files = os.listdir(datasetPath)
for fileName in files:
    if not fileName.endswith('.jpg') or not os.path.isfile(os.path.join(datasetPath,fileName)):
        continue

    img = cv2.imread(os.path.join(datasetPath,fileName))

    image = ET.SubElement(images, 'image', file=fileName, width=str(img.shape[0]), height=str(img.shape[1]))

    try:
        # Detecta a face = left, top, right, and bottom
        detections = detector.detect(img)[0]
    except:
        detections = [0,0,0,0]
        
    box = ET.SubElement(image, 'box', top=str_trunc(detections[1]), left=str_trunc(detections[0]), width=str_trunc(detections[2]-detections[0]), height=str_trunc(detections[3]-detections[1]))

    # Lê arquivo .pts com pontos do rosto
    with open(os.path.join(datasetPath, "landmarks", fileName.replace('.jpg','.pts'))) as f:
        rows = [rows.strip() for rows in f]
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    contadorLandmark = 0
    for coords in coords_set:
        # Para cada ponto dos 68 landmarks, criar elemento "part" no xml
        part = ET.SubElement(box, 'part', name=str(contadorLandmark), x=str_trunc(coords[0]), y=str_trunc(coords[1]))
        contadorLandmark +=1

# Salva xml indentado
tree = ET.ElementTree(root)
ET.indent(tree, '  ')
tree.write('training_with_face_landmarks.xml', encoding="utf-8", xml_declaration=True)