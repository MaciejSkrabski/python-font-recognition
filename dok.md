# PUMA - projekt

Klasyfikator fontów z obrazu

Tomasz Sitko, Maciej Skrabski, Łukasz Wójcik

## Opis

Zaklasyfikowanie fonta jakim napisane jest słowo na obrazku.

Do wyboru są trzy fonty:

- Lato-Regular,
- LiberationSans-Regular
- LiberationSerif-Regular

Porównywane metody klasyfikacji:

- SVM (Grid Search)
- CNN (dwie wartswy konwolucyjne z warstwami pooling między nimi, jedna warstwa _fully connected_)

## Zbiór danych

Wygenerowany przez nas zbiór 600 obrazów _jpg_ w skali szarości oraz odpowiadający im plik _csv._ wraz z danymi dla danego obrazka. Każdy obrazek ma wymiary 300x80. Zawiera w sobie jedno losowo wybrane słowo (ze słownika _csv_) występujące tylko raz na cały zbiór. Słowo zapisane jest jednym z trzech fontów. Po dwieście obrazków na każdy font. Napis ma losowy rozmiar fonta z przedziału [60, 70]. Tekst na obrazie jest losowo obrócony o losową wartość z przedziału [-2, 2] stopnie. Kolumny w _csv_:
` | id     | słowo     | etykieta fonta | `

## Przykładowe obrazy

Lato-Regular:

![abdicative.jpg](./dataset/abdicative.jpg "abdicative.jpg")

LiberationSerif-Regular:

![abdicative.jpg](./dataset/crystalliferous.jpg "crystalliferous.jpg")

LiberationSans-Regular:

![metallization.jpg](./dataset/metallization.jpg "metallization.jpg")

---

## Uczenie

W obu przypadkach:
> `treningowy 80 / 20 testowy`

### SVM

```py
SVM

split data
svc
grid search, wielowątkowo - czas start
GridSearchCV:
 Najlepsze parametry:  {'C': 0.1, 'degree': 2, 'gamma': 0.03125, 'kernel': 'poly'}
 Sredni wynik kross-walidacji: 0.8145833333333333
 Dokladnosc na zbiorze testowym: 0.8416666666666667
 Klasyfikator nauczono w czasie  963.9532241821289


randomized search cv
randomized search cv fit, rozpoczęcie uczenia - czas start
RandomizedSearchCV:
 Najlepsze parametry:  {'kernel': 'poly', 'gamma': 5, 'degree': 5, 'C': 0.1}
 Sredni wynik kross-walidacji: 0.8104166666666667
 Dokladnosc na zbiorze testowym: 0.8416666666666667
 Klasyfikator nauczono w czasie 192.69022727012634

```

Ponad 16 minut na nauczenie klasyfikatora SVM. Algorytm działał wielowątkowo.
Dokładnośc na poziomie 84%.

---

### CNN

```py
Convolution layer:
W x H x D -> Wc x Hc x Dc   gdzie:

```

$$ W_c = \frac{W-F+2P}{S}+1 $$
$$ H_c = \frac{H-F+2P}{S}+1 $$
$$ D_c = K$$

```py
gdzie:
W - width
H- height
D - depth
K - liczba filtrów
F - rozmiar kernela
P - zero padding
```

```py
Pooling layer:
W x H x D -> Wp x Hp x Dp   gdzie:

```

$$ W_p = \frac{W-F}{S}+1 $$
$$ H_p = \frac{H-F}{S}+1 $$
$$ D_p = D $$

```py
gdzie:
W - width
H- height
D - depth
K - liczba filtrów
F - rozmiar kernela
P - zero padding
```

```py
Model:

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
```

```py
rozpoczęcie uczenia - czas start
Epoka: 1, 20, wartość funkcji kosztu: 4.779
Epoka: 1, 40, wartość funkcji kosztu: 9.213
Epoka: 1, 60, wartość funkcji kosztu: 12.849
(...)
Epoka: 7, 80, wartość funkcji kosztu: 0.018
Epoka: 8, 20, wartość funkcji kosztu: 0.003
Epoka: 8, 40, wartość funkcji kosztu: 0.006
(...)
Epoka: 12, 40, wartość funkcji kosztu: 0.001
Epoka: 12, 60, wartość funkcji kosztu: 0.001
Epoka: 12, 80, wartość funkcji kosztu: 0.002
Uczenie zakończone w czasie 15.969974040985107
```

```py
model = Font_CNN().to(device)...
Sprawdzenie zgodności ze zbiorem testowym - czas start
Dokładność klasyfikacji na 120 egzemplarzach zbioru testowego wynosi: 96.67%
Dokładność klasyfikacji dla klasy Lato-Regular wynosi 94.44%.
Dokładność klasyfikacji dla klasy LiberationSans-Regular wynosi 95.00%.
Dokładność klasyfikacji dla klasy LiberationSerif-Regular wynosi 100.00%.
Zgodność ze zbiorem testowym sprawdzono w czasie  0.34691405296325684

```
