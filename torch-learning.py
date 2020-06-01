#%%
from PIL import Image
import numpy as np


class Img:
    def __init__(self, row):
        def to_numpy(word):
            # loading image to numpy array
            pic = Image.open("dataset/"+word+".jpg")
            pix = pic.getdata(0)
            return np.array(pix)/255 # converting to float in range [0, 1]
    
        # the object has id, a word depicted on it's image, a label
        # with the font used to write the word on the image and data,
        # which is a numpy array of pixels' brightnesses in range[0,1].
        self.id = int(row[0])
        self.word = row[1]
        self.label = row[2]
        self.data = to_numpy(self.word)

    

#%%
from csv import reader

list_of_images=[]
list_of_rows=[]
with open("dataset/dataset.csv") as csvfile:
    r = reader(csvfile)
    next(r)
    for row in r:
        list_of_rows.append(row)

# print(l[:5], l[-5:], len(l)) # testgi
#%%
for row in list_of_rows:
    list_of_images.append(Img(row))
#test
row = list_of_images[0]
row2 = list_of_images[599]
print(row.id, row.word, row.label, row.data, row.data.shape)
print(row2.id, row2.word, row2.label, row2.data, row2.data.shape)
# %%
X = []
y = []
list_of_images[0]
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

# %%

# %%
