# Navodila

## I.) Učenje in testiranje modelov
### 1. datareader.py
Za branje/odpiranje slik - ne spreminjaj.
### 2. model.py
Za arhitekturo konvolucijske mreže - spremeni le funkciji ```init``` in ```forward```.
### 3. train.py
Za učenje modela in hkratno preverjanje uspešnosti na validacijski množici. Po potrebi spremeni način vrednotenja modela (trenutno je končna verjetnost za pripadnost razredu 0/1 definirana kot povprečje verjetnosti 10 centralnih rezin).
### 4. test.py
Za testiranje modela na testni množici. Ne spreminjaj ```batch_size=10``` in ```shuffle=False```.
### 5. run.py
Primer uporabe učenja in testiranja na Marvinu. Za lokalno uporabo spremeni ```main_path_to_data```.

## II.) Interpretacija modelov

### 1. visualize_filters.py
Vizualizacija filtrov naučenega modela.
### 2. visualize_convolutions.py
Za vizualizacijo rezultatov konvolucij vhodne slike na različnih globinah mreže. Po
### 3. visualize_saliency.py
Mapa izpostavljenosti oz. saliency map za doprinos k verjetnosti za hudo okužbo. 