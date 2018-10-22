import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import pprint as pp
from darkflow.net.build import TFNet
import cv2


options = {"model": "cfg/yolo_custom.cfg", 
           "load": "bin/yolo.weights",
	   "labels": "labels.txt",
	   "summary": "DIR/",
           "batch": 8,
           "epoch": 100,
           "gpu": 1.0,
           "train": True,
           "annotation": "annotations/",
           "dataset": "images/"}


tfnet = TFNet(options)

tfnet.train()

tfnet.savepb()

options = {"model": "cfg/yolo_custom.cfg",
           "load": -1,
           "gpu": 1.0}

tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()

original_img = cv2.imread("sample_img/test_image1.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(original_img)

def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage


fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(boxing(original_img, results))

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

for i in range(5):
    original_img = cv2.imread("sample_img/test_image" + str(i+1) + ".jpg")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet2.return_predict(original_img)
    print(results)
    ax[ceil(i/3)-1, i%3].imshow(boxing(original_img, results))

plt.show()
