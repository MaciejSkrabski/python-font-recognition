# %%
from PIL import Image
import numpy as np
import torch
import torchvision
import sys
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
        pic = pic[:, :, 0]/255
        return pic  # converting to float in range [0, 1]

    def to_numpy(self):
        return self.data.numpy()

    def __init__(self, row):
        # the object has id, a word depicted on it's image, a label
        # with the font used to write the word on the image and data,
        # which is a numpy array of pixels' brightnesses in range[0,1].
        self.id = int(row[0])
        self.word = row[1]
        self.label = row[2]
        self.data = self.to_tensor(self.img_to_numpy(self.word))


# %%


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
# print(l[:5], l[-5:], len(l)) # test
# %%


print(row2.id, row2.word, row2.label, "\n", row2.data)

print("\n\n", np.shape(row2.to_numpy()))

# display example images from dataset
plt.figure(figsize=(5, 5))
plt.subplot(221), plt.imshow(row.to_numpy(), cmap='gray')
plt.subplot(222), plt.imshow(row2.to_numpy(), cmap='gray')
plt.subplot(223), plt.imshow(images[100].to_numpy(), cmap='gray')
plt.subplot(224), plt.imshow(images[150].to_numpy(), cmap='gray')
plt.show()


# %%
