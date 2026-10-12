# Symulacje stochastyczne – listy zadań laboratoryjnych

Notebooki Jupyter z rozwiązaniami sześciu list zadań laboratoryjnych z kursu poświęconego metodom Monte Carlo i symulacjom stochastycznym, napisane w Pythonie.

## Struktura repozytorium

Każda lista znajduje się w osobnym folderze:

| Folder | Tematyka |
|--------|----------|
| `lista_1/` | Zadania wprowadzające: generowanie zmiennych losowych, gry z monetą i kośćmi, prawdopodobieństwo bankructwa, problem kolekcjonera |
| `lista_2/` | Metoda odwracania dystrybuanty, dystrybuanta empiryczna, numeryczne odwracanie dystrybuanty, wykresy kwantylowe (QQ-plot), weryfikacja generatorów, VaR jako kwantyl |
| `lista_3/` | Metoda akceptacji-odrzucenia, metoda Boxa-Mullera (zwykła i biegunowa), dekompozycja Cholesky'ego, całkowanie metodą Monte Carlo |
| `lista_4/` | Proces Poissona: generowanie trajektorii, kompensowany proces Poissona, generatory rozkładu Poissona |
| `lista_5/` | Niejednorodny proces Poissona (thinning), złożony proces Poissona, prosty proces ryzyka |
| `lista_6/` | Proces ryzyka i ruina paryska, mieszany proces Poissona, ruch Browna, procesy Lévy'ego, geometryczny ruch Browna |

Foldery `lista_2/` i `lista_3/` zawierają również notebooki z opcjonalnymi zadaniami dodatkowymi (`lista2_zad_dodatkowe.ipynb`, `Lista3_dodatkowe.ipynb`).

## Wymagania

- Python 3.9+
- Jupyter Notebook lub JupyterLab
- `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`
