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

Wygenerowany przez nas zbiór 600 obrazów _jpg_ oraz odpowiadający im plik _csv._ wraz z danymi dla danego obrazka. Każdy obrazek ma wymiary 300x80. Zawiera w sobie jedno losowo wybrane słowo (ze słownika _csv_) występujące tylko raz na cały zbiór. Słowo zapisane jest jednym z trzech fontów. Po dwieście obrazków na każdy font. Napis ma losowy rozmiar fonta z przedziału [60, 70]. Tekst na obrazie jest losowo obrócony o losową wartość z przedziału [-2, 2] stopnie. Kolumny w _csv_:
` | id     | słowo     | etykieta fonta | `
