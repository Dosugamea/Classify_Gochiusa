# Classify_Gochiusa
My first machine learning   
  
Kerasを用いてごちうさのキャラを機械学習してみたものです。  
Machine learning of character from "Is the order a rabbit?" by keras  
  
imgに画像を入れて実行したらdetectに枠を付けた結果が出ます。  
this can detect learned character and separate to images  
  
とりあえず　個人利用では使えるかもしれないという検出率です。  
really bad program but possibly useable to personal usage  

やってみたかっただけなので僕はどう動いているのかさっぱりわかりません。  
参考にはしない方がいいです。  
I don't know how this program works.  
Don't recommend to use this to learning.
  
## Learn info
I used separeted images by OpenCV with lbpcascade_animeface.

vgg16_gochiusa_fine
[Base]
> Epoch 20
> BatchSize 32
> ShapeSize 64
> val_acc 0.97
[Used Pictures]
> aoyama x54
> chino x2239
> chiya x337
> cocoa x777
> maya x162
> megu x139
  
vgg16_gochiusa_fine2
[Base]
> Epoch 30
> BatchSize 32
> ShapeSize 128
> val_acc 0.94
[Used Pictures]
> aoyama x54
> chino x2239
> chiya x337
> cocoa x777
> maya x162
> megu x139
> other x688
  
##Bibliography
VGG16を転移学習させて「まどか☆マギカ」のキャラを見分ける  
https://qiita.com/God_KonaBanana/items/2cf829172087d2423f58  
機械学習でNEW GAME!のキャラを判別してみた  
https://konnyaku256.com/tweepy/newgame  
Kerasでアニメキャラの顔認識  
https://qiita.com/sartan123/items/6abd5e0d814029f9c7d5  
Keras(Tensorflow)の学習済みモデルのFine-tuningで少ない画像からごちうさのキャラクターを分類する分類モデルを作成する  
https://qiita.com/kazuki_hayakawa/items/c93a21313ccbd235b82b