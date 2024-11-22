!pip install face_recognition

from google.colab import drive
import os

drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/face_recog') #切換該目錄
os.listdir()

# New Section

import face_recognition
import matplotlib.pyplot as plt
test = "20241122_133339.jpg"
img = face_recognition.load_image_file(test)
en = face_recognition.face_encodings(img)[0]
print("化簡過後:", en)
plt.imshow(img)

import glob
fs = glob.glob("*.jpg") + glob.glob("*.png")
names = []
encodings = []
for f in fs:
    img = face_recognition.load_image_file(f)
    en = face_recognition.face_encodings(img)[0]
    n = f.split(".")[0]
    names.append(n)
    encodings.append(en)
print ("名字:",names)
print ("簡化:",encodings)
