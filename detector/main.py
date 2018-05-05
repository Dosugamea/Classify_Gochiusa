from keras.models import model_from_json
import numpy as np
import os,random,cv2,shutil
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from copy import deepcopy

colors = {
    'aoyama':[255,242,230],
    'chino':[193,255,255],
    'chiya':[194,255,214],
    'cocoa':[255,204,213],
    'maya':[204,213,255],
    'megu':[255,153,221],
    'rize':[221,153,255],
    'syaro':[255,255,204],
    'other':[0,0,0]
}

#Read datas
print("detect mode? None(1) or 2")
mode = input('>>')
file_name='vgg16_gochiusa_fine'+mode
print("separate files? y/n(None)")
spl = input('>>')

if "2" in file_name:
    md_size = 128
    label=['aoyama','chino','chiya','cocoa','maya','megu','rize','syaro','other']
else:
    md_size = 64
    label=['aoyama','chino','chiya','cocoa','maya','megu','rize','syaro']
json_string=open(file_name+'.json').read()
model=model_from_json(json_string)
model.load_weights(file_name+'.h5')
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#顔検出するやつ
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
input_dir = "img"

if "2" in file_name:
    output_dir = "detected2"
else:
    output_dir = "detected"
internal_dir = "faces"

for delf in os.listdir(output_dir):
    if ".db" not in delf:
        os.remove(output_dir+"/"+delf)
files = os.listdir(input_dir)
ix = 0
#ファイル数分繰り返す
for file in files:
    print("Processing %s"%(file))
    img = cv2.imread(input_dir+"/"+file)
    imgX = deepcopy(img)
    faces = face_cascade.detectMultiScale(img,minNeighbors=10)
    if len(faces) > 0:
        for delf in os.listdir(internal_dir):
            if ".db" not in delf:
                os.remove(internal_dir+"/"+delf)
        names = []
        #顔数分繰り返す(準備)
        for i,(x,y,w,h) in enumerate(faces):
            #顔画像を作って
            face_image = img[y:y+h, x:x+w]
            output_path = os.path.join(internal_dir,'{0}.jpg'.format(i))
            cv2.imwrite(output_path,face_image)
            #顔画像を読み込んで推測
            temp_img=load_img("./"+internal_dir+"/"+str(i)+".jpg",target_size=(md_size,md_size))
            temp_img_array=img_to_array(temp_img)
            temp_img_array=temp_img_array.astype('float32')/255.0
            temp_img_array=temp_img_array.reshape((1,md_size,md_size,3))
            img_pred=model.predict(temp_img_array)
            cls = colors[label[np.argmax(img_pred)]]
            cv2.rectangle(imgX, (x, y), (x+w, y+h), (cls[2],cls[1],cls[0]), 3)
            '''
            cv2.putText(img, label[np.argmax(img_pred)], (x+w-60, y+h+30),
               cv2.FONT_HERSHEY_PLAIN, 3,
               (0, 0, 0), 1, cv2.LINE_AA)
            '''
            print("%s.jpg"%(i))
            print(img_pred)
            print("Maybe: "+label[np.argmax(img_pred)])
            if spl == "y":
                cv2.imwrite("./splits/"+label[np.argmax(img_pred)]+"/"+str(ix)+".png",face_image)
                ix+=1
        cv2.imwrite(output_dir+"/d_"+file,imgX)
print("Complete")