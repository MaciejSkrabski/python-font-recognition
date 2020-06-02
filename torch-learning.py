# %%
from PIL import Image
import numpy as np
import torch
import torchvision

from csv import reader
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
print("CUDA available?", is_cuda)


class Img:
    def to_tensor(self, numpyarr):
        return torch.from_numpy(numpyarr)

    def img_to_numpy(self, word):
        # loading image to numpy array
        pic = np.asarray(Image.open("dataset/"+word+".jpg"))
        pic = pic.astype('float32')
        pic = pic[:, :, 0]/255  # normalization
        mean = pic.mean()
        # global centering of normalized pixels
        pic = pic - mean
        return pic

    def to_numpy(self):
        return self.data.numpy()

    def __init__(self, row):
        # the object has id, a word depicted on it's image, a label
        # with the font used to write the word on the image and data,
        # which is a numpy array of pixels' brightnesses in range[0,1].
        self.id = int(row[0])
        self.word = row[1]
        self.label = row[2]
        self.data = self.img_to_numpy(self.word)


# %%
# data loading

images = []
rows = []
with open("dataset/dataset.csv") as csvfile:
    r = reader(csvfile)
    next(r)
    for row in r:
        rows.append(row)

for row in rows[:200]:
    images.append(Img(row))
# test
row = images[0]
row2 = images[50]
# %%


print(row2.id, row2.word, row2.label, "\n", row2.data)

print("\n\n", np.shape(row2.data))

# display example images from dataset
plt.figure(figsize=(5, 5))
plt.subplot(221), plt.imshow(row.data, cmap='gray')
plt.subplot(222), plt.imshow(row2.data, cmap='gray')
plt.subplot(223), plt.imshow(images[100].data, cmap='gray')
plt.subplot(224), plt.imshow(images[150].data, cmap='gray')
plt.show()


# %%
X = []
y = []
for image in images:
    X.append(image.data)
    # converting label to target values
    if image.label == "Lato-Regular":
        y.append(0)
    elif image.label == "LiberationSans-Regular":
        y.append(1)
    elif image.label == "LiberationSerif-Regular":
        y.append(2)
    else:
        print("SUM TING WONG!", image.label)
        break
X = np.asarray(X)
y = np.asarray(y)


# %%
# podział na zbiór testowy i treningowy
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(2)  # freezing seed for predictability
    p = np.random.permutation(len(a))
    return a[p], b[p]


split = int(np.round(len(X)*.8))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
# simultanous shuffling of two arrays
X_train, y_train = unison_shuffled_copies(X_train, y_train)

# conversion to torch tensor
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
# %%
