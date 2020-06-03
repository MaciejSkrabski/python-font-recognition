# %%

import pandas as pd
import numpy as np

from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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

# %%
for row in list_of_rows:
    list_of_images.append(Img(row))

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
X = np.asarray(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# do sprawdzania dla pojedynczego jadra linear
# model = svm.SVC(kernel='linear', C=10).fit(X_train,
#                                             y_train)

# do sprawdzenia dla rbf
# model = svm.SVC(kernel='rbf', gamma=0.7, C=10).fit(X_train,
#                                                   y_train)

# do sprawdzenia dla poly
# model = svm.SVC(kernel='poly', degree=5, gamma=0.03, C=0.1).fit(X_train,
#                                                                  y_train)

# wypisanie dokladnosci na zbiorze testowym
# print(model.score(X_test, y_test))

parameters = {'kernel': ('linear', 'rbf', 'poly'),
              'C': [0.1, 1, 10, 100],
              'gamma': [2**-5, 5],
              'degree': [2, 3, 4, 5]}

svc = svm.SVC()
grid_search_cv = GridSearchCV(svc, parameters, cv=4)
grid_search_cv.fit(X_train, y_train)

print("GridSearchCV:")
print(" Najlepsze parametry: ", grid_search_cv.best_params_)
print(" Sredni wynik kross-walidacji:", grid_search_cv.best_score_)
print(" Dokladnosc na zbiorze testowym:", grid_search_cv.score(X_test, y_test))
print("\n")

randomized_search_cv = RandomizedSearchCV(svc, parameters, cv=4, random_state=3)
randomized_search_cv.fit(X_train, y_train)

print("RandomizedSearchCV:")
print(" Najlepsze parametry: ", randomized_search_cv.best_params_)
print(" Sredni wynik kross-walidacji:", randomized_search_cv.best_score_)
print(" Dokladnosc na zbiorze testowym:", randomized_search_cv.score(X_test, y_test))
