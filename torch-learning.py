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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)


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
X_train, y_train = X[:split], y[:split]


# simultaneous shuffling of two arrays
X_train, y_train = unison_shuffled_copies(X_train, y_train)

# conversion to torch tensor
# training set
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 18, kernel_size=3, stride=1, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(18*20*75, 3)  # zwykły iloczyn wymiarów

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))  # wymiary pozostały niezmienne
        xb = self.pooling(xb)  # wymiary/2
        xb = F.relu(self.conv2(xb))  # wymiary pozostały niezmienne
        xb = self.pooling(xb)  # wymiary/2
        # "spłaszczenie" danych przed wejściem do warstw w pełni połączonych
        xb = xb.view(-1, 18*20*75)

        xb = self.fc(xb)  # fully connected - zlinearyzowanie pod etykiety

        return xb

# %%
bs = 5  # batch size

xb = X_train[0:bs]  # a mini-batch from x
xb = xb.view(bs, 1, 80, 300)  # batch size, dim, h, w
# Tworzymy obiekt klasy CNN2D i jeśli to możliwe,
# przenosimy go na kartę graficzną (możliwe tylko dla kart
# wspierających obliczenia CUDA!)
model = Font_CNN()

model.to(device)


# funkcja spadku
loss = nn.CrossEntropyLoss()

# Optymalizator będzie propagował błąd podczas procesu uczenia.
# Potrzebne są mu do tego parametry modelu,
# a także współczynnik uczenia, który ustawiamy na 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
all_elems = len(X_train)


for epoch in range(n_epochs):
    # startujemy od wartości
    # funkcji kosztu wynoszącej 0
    # shuffle data every epoch!!!
    xs, ys = unison_shuffled_copies(X_train, y_train)
    cumulative_loss = 0.0
    batch = 5
    j = 0
    while (j*batch <= all_elems-batch):
        inputs, labels = xs[
            j*batch:(j+1)*batch
            ].to(device), ys[
                j*batch:(j+1)*batch
                ].to(device)
        inputs = inputs.view(batch, 1, 80, 300)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss_value = loss(outputs, torch.max(labels, 1)[1])
        print(torch.max(labels, 1)[1])
        loss_value.backward()
        optimizer.step()
        cumulative_loss += loss_value.item()
        if j % 5 == 4:
            print(
                'Epoka: {}, {}, wartość funkcji kosztu: {:.3f}'.
                format(epoch + 1,
                       j + 1,
                       cumulative_loss / 5
                       ))
        j += 1


print('Uczenie zakończone')

# %%
# zapis do pliku
torch.save(model.state_dict(), './Fonts_CNN.pth')


# %%
model = Font_CNN().to(device)
model.load_state_dict(torch.load('./Fonts_CNN.pth'))


correct_number, total_number = 0, 0
class_correct, class_total = np.zeros(3), np.zeros(3)
with torch.no_grad():
    batch = 5
    j = 0
    test_elems = len(X_test)
    while (j*batch <= test_elems-batch):
        inputs, labels = X_test[
            j*batch:(j+1)*batch
            ].to(device), torch.max(y_test[
                j*batch:(j+1)*batch
                ].to(device), 1)[1]
        inputs = inputs.view(batch, 1, 80, 300)

        outputs = model(inputs)
        prediction = torch.max(outputs, 1)[1]  # 
        total_number += labels.size(0)

        check_prediction = (prediction == labels)
        # print(prediction, labels, "\n\n")

        correct_number += check_prediction.sum().item()
        # Dokonujemy spłaszczenia wektora.
        check_prediction = check_prediction.squeeze()

        for i in range(5):
            # Sprawdźmy jego etykietę.
            label = labels[i]
            # Jeżeli sieć dobrze przewidziała wynik, liczba prawidłowo
            # zaklasyfikowanych przypadków tej kategorii wzrośnie o 1,
            # w przeciwnym wypadku o 0.
            class_correct[label] += check_prediction[i].item()
            # Wczytaliśmy jeden przypadek danej klasy, więc dodajemy go
            #  do ogólnego wyniku.
            class_total[label] += 1

        j += 1

print('Dokładność klasyfikacji na {} egzemplarzach zbioru testowego wynosi: '
      '{:.2f}%'.format(test_elems, 100.0 * correct_number / total_number))
classes = ("Lato-Regular", "LiberationSans-Regular", "LiberationSerif-Regular")
for i in range(3):
    print(
        'Dokładność klasyfikacji dla klasy {} \
wynosi {:.2f}%.'.format(classes[i],
                        100.0 * class_correct[i] / class_total[i]))
