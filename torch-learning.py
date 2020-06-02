# %%
from PIL import Image
import numpy as np
import torch
from csv import reader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2)
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

for row in rows:
    images.append(Img(row))
# test
row = images[0]
row2 = images[50]
# %%


print(row2.id, row2.word, row2.label, "\n")

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
        y.append([1., 0., 0.])
    elif image.label == "LiberationSans-Regular":
        y.append([0., 1., 0.])
    elif image.label == "LiberationSerif-Regular":
        y.append([0., 0., 1.])
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
X_test, y_test = X[split:], y[split:]
X_t, y_t = X[:split], y[:split]
X_train, y_train = X_t[:split], y_t[:split]
X_valid, y_valid = X_t[split:], y_t[split:]
# simultaneous shuffling of two arrays
X_train, y_train = unison_shuffled_copies(X_train, y_train)

# conversion to torch tensor
# training set
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# validation set
X_valid = torch.from_numpy(X_valid)
y_valid = torch.from_numpy(y_valid)

# testing set
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# %%
# defining cnn

# https://towardsdatascience.com/understand-the-architecture-of-cnn-90a25e244c7
# Convolution layer:
# the filters/kernels are small and dragged on the image one pixel at a time.
# The zero-padding value is chosen so that the width and height of the
# input volume are not changed at the output.
# (W-F+2*P)/S + 1 == W
# In our case: 298+2*P == 300   =>   P == 1
# (H-F+2*P)/S + 1 == H
# In our case: 78+2*P == 300   =>   P == 1

# Pooling layer:
# reduces size of the picture preserving its features.
# This helps with avoiding overfitting.
# F = 2, S = 2


class Font_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, xb):
        # xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 3)
        return xb

bs = 64  # batch size

xb = X_train[0:bs]  # a mini-batch from x
# Tworzymy obiekt klasy CNN2D i jeśli to możliwe,
# przenosimy go na kartę graficzną (możliwe tylko dla kart
# wspierających obliczenia CUDA!)
model = Font_CNN()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
model.to(device)
print(device)
model(xb)

# %%

# funkcja spadku
loss = nn.CrossEntropyLoss()

# Optymalizator będzie propagował błąd podczas procesu uczenia.
# Potrzebne są mu do tego parametry modelu,
# a także współczynnik uczenia, który ustawiamy na 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
n_epochs = 5

for epoch in range(n_epochs):
    # startujemy od wartości
    # funkcji kosztu wynoszącej 0
    cumulative_loss = 0.0

    for i, data in enumerate(X_train, 0):
        print(data[0])
        # przenosimy dane na kartę graficzną, jeśli to możliwe
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zerujemy gradienty
        optimizer.zero_grad()

        # Wczytujemy dane do modelu
        outputs = model(inputs)
        # Wyliczamy wartość funkcji kosztu,
        # czyli porównujemy wyjście z sieci z etykietami
        loss_value = loss(outputs, labels)
        # Dokonujemy propagacji błędu i optymalizujemy parametry
        loss_value.backward()
        optimizer.step()
        
        # Drukujemy cząstkowe wyniki co 2000 mini-paczek
        cumulative_loss += loss_value.item()
        if i % 2000 == 1999:
            print('Epoka: {}, {}, wartość funkcji kosztu: {:.3f}'.format(epoch + 1, i + 1, cumulative_loss / 2000))
            cumulative_loss = 0.0

print('Uczenie zakończone')

# %%
