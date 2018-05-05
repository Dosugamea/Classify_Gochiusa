from keras.models import model_from_json
import numpy as np
import os,random,cv2,shutil
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

'''
Based
https://github.com/Kisaragi-2/VGG16/blob/master/madoka_magica/vgg16_madomagi_predict.py
'''

#学習したデータを読み込む
file_name='vgg16_gochiusa_fine'
label=['aoyama','chino','chiya','cocoa','maya','megu','rize','syaro']
json_string=open(file_name+'.json').read()
model=model_from_json(json_string)
model.load_weights(file_name+'.h5')
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#顔検出するやつ
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
output_dir = 'faces'

while True:
    try:
        shutil.rmtree("./"+output_dir)
        os.makedirs(output_dir)
    except:
        pass
    print("Input FileName")
    lf = input('>>')
    img = cv2.imread("./data/display/"+lf)
    faces = face_cascade.detectMultiScale(img,minNeighbors=10)
    print(faces)
    if len(faces) > 0:
        for i,(x,y,w,h) in enumerate(faces):
            face_image = img[y:y+h, x:x+w]
            output_path = os.path.join(output_dir,'{0}.jpg'.format(i))
            cv2.imwrite(output_path,face_image)
            temp_img=load_img("./faces/"+str(i)+".jpg",target_size=(64,64))
            #Images normalization
            temp_img_array=img_to_array(temp_img)
            temp_img_array=temp_img_array.astype('float32')/255.0
            temp_img_array=temp_img_array.reshape((1,64,64,3))
            #predict image
            img_pred=model.predict(temp_img_array)
            #print(img_pred)
            print(str(i)+".jpg")
            print("MAX: "+label[np.argmax(img_pred)])
            print("MIN: "+label[np.argmin(img_pred)])