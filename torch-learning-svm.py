# %%

import pandas as pd
import numpy as np

from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc


class Img:
    def __init__(self, row):
        def to_numpy(word):
            # loading image to numpy array
            pic = Image.open("dataset/" + word + ".jpg")
            pix = pic.getdata(0)
            return np.array(pix) / 255  # converting to float in range [0, 1]

        # the object has id, a word depicted on it's image, a label
        # with the font used to write the word on the image and data,
        # which is a numpy array of pixels' brightnesses in range[0,1].
        self.id = int(row[0])
        self.word = row[1]
        self.label = row[2]
        self.data = to_numpy(self.word)


# %%
from csv import reader

list_of_images = []
list_of_rows = []
with open("dataset/dataset.csv") as csvfile:
    r = reader(csvfile)
    next(r)
    for row in r:
        list_of_rows.append(row)

# print(l[:5], l[-5:], len(l)) # testgi
# %%
for row in list_of_rows:
    list_of_images.append(Img(row))
# test
row = list_of_images[0]
row2 = list_of_images[599]
# print(row.id, row.word, row.label, row.data, row.data.shape)
# print(row2.id, row2.word, row2.label, row2.data, row2.data.shape)
# %%
X = []
y = []
for obj in list_of_images:
    X.append(obj.data)
    if obj.label == "Lato-Regular":
        y.append(0)
    elif obj.label == "LiberationSans-Regular":
        y.append(1)
    elif obj.label == "LiberationSerif-Regular":
        y.append(2)
    else:
        print("SUM TING WONG!", obj.label)
        break
print(list_of_images[0].data.shape)
X = np.asarray(X)
print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=2)

print("test2")

# nsamples, nx, ny = y_train.

# nsamples, nx, ny = train_dataset.shape
# d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

parameters = {'kernel': ('linear', 'rbf', 'poly'),
              'C': [1, 100],  # [1,10,100,1000],
              'gamma': [0.001,0.01,1,10],  # [2**-5,5],
              'degree': [1,2,3]}  # [2**-5,5]}

svc = svm.SVC()

print("\n\n")

# grid_search_cv = GridSearchCV(svc, parameters, cv = 4)
# grid_search_cv.fit(x_train, y_train)
# print("test3")
# print("GridSearchCV:")
# print("Najlepsze: ", grid_search_cv.best_params_)
# print("Najlepsze parametry: ", grid_search_cv.best_params_)
# print("Dokladnosc klasyfikacji dla zbioru treningowego:", grid_search_cv.best_score_)
# print("Dokladnosc klasyfikacji dla zbioru testowego:", grid_search_cv.score(x_test,y_test))
#
# print("\n\n")
#
randomized_search_cv = RandomizedSearchCV(svc, parameters, cv=4)
randomized_search_cv.fit(x_train, y_train)

print("RandomizedSearchCV:")
# print("Najlepsze parametry: ", randomized_search_cv.best_params_)
# print("Dokladnosc klasyfikacji dla zbioru treningowego:", randomized_search_cv.best_score_)
# print("Dokladnosc klasyfikacji dla zbioru testowego:", randomized_search_cv.score(x_test, y_test))
