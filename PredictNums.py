import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import cv2
from dataset import load_hoda
#لود دیتا ها
X_train, y_train, X_test, y_test = load_hoda()
#تعداد همسایه ها
model=KNeighborsClassifier(n_neighbors=3)
#train model
model.fit(X_train,y_train)
#predict on test data
predicted=model.predict(X_test)
print("دقت مدل در پیش بینی داده های تست")
print(model.score(X_test,y_test))



#نمونه از داده خارجی
img=cv2.imread('persian-digits.jpg')
#تبدیل gray
gray=cv2.imread('persian-digits.jpg',0)
#تبدیل فقط  به 0 و 255
gray[gray>127.5]=255
gray[gray<127.5]=0
gray=255-gray



#OCR
num_labels,labels,stats,centroid=cv2.connectedComponentsWithStats(gray,connectivity=8)
img=cv2.imread('persian-digits.jpg')

for i in range(1,num_labels):
    x,y,w,h,_=stats[i]
    if w>5 and h>5:
        image_gray=gray[y:y+h,x:x+w]
        img_resize=cv2.resize(image_gray,(5,5))
        img_reshape=img_resize.reshape(1,25)
        result=model.predict(img_reshape)
        print(result)
        cv2.putText(img,str(result),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
#چون توی cv bgr داریم تبدیل شد اینجا به rgb
plt.imshow(img[:,:,::-1])
plt.show()