from dataset import load_hoda
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


x_train,y_train,x_test,y_test = load_hoda()


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

#predict new number
pred_classes=model.predict(x_test)
print(pred_classes)