# %%
from PIL import Image
import numpy as np
import torch
import torchvision
import sys
from csv import reader


print("CUDA available?", torch.cuda.is_available())


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


list_of_images = []
list_of_rows = []
with open("dataset/dataset.csv") as csvfile:
    r = reader(csvfile)
    next(r)
    for row in r:
        list_of_rows.append(row)

# print(l[:5], l[-5:], len(l)) # test
# %%


for row in list_of_rows[:3]:
    list_of_images.append(Img(row))
# test
row = list_of_images[0]
row2 = list_of_images[2]
print(row2.id, row2.word, row2.label, "\n", row2.data)

print("\n\n", np.shape(row2.to_numpy()))

# np.savetxt(sys.stdout, (row2.to_numpy())[:1], '%.001e')

# for idx, val in enumerate(row.data):
#     for inner, oy in enumerate(val):
#         if oy!=1:
#             print(idx, inner, oy)


# %%
