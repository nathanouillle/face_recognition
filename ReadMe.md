# Projet de Machine Learning

Ce projet est un exemple d'application d'un modèle de Machine Learning pour reconnaître des visages.

## Prérequis

* Python 3.8 ou supérieur
* Pytorch 1.10 ou supérieur
* Matplotlib 3.5 ou supérieur

## Installation

1. Installez les dépendances :

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
```

2. Clonez le dépôt :

```
git clone https://github.com/nathanouillle/face_recognition.git
```

3. Récupérez les données :

Récupérez les *test_images* et *train_images* sur http://www.duffner-net.de et sauvegardez les dossiers dans le dossier **face_recognition**


## Utilisation

Pour exécuter le projet, ouvrez les notebooks Jupyter dans le répertoire du projet.

* Le notebook `main.ipynb` contient le déroulement de la construction, entraînement et vérification du modèle de détection de visages.

* `net.py` est l'architecture du réseau de neurones.

* `model.pth` est le réseau de neurones entrainé pour reconnaitre les visages.

* Le notebook `sliding_window.ipynb` contient le code de l'utilisation du dit modèle sur une image.

* Le dossier `images_sliding_window` contient des exemples d'images sur lequels le sliding window peut être utilisé.

## Explication du code

Le code du notebook `main.ipynb` est divisé en trois parties principales itératives :

* **Importation des données**
* **Entraînement du modèle**
* **Évaluation du modèle**

Dans la première partie, les données sont importées, transformées et séparées à partir des dossier test_images et train_images.

Dans la deuxième partie, le modèle est entraîné sur les données.

Dans la troisième partie, le modèle est évalué sur un ensemble de données de test.

---

Le code du notebook `sliding_window.ipynb` est divisé en trois parties principales :

* **Choix des paramètres**
* **Sliding Window**
* **NMS**

Dans la première partie, les paramètres pertinents du sliding window sont définis.

Dans la deuxième partie, le l'importation du modèle, transformation de l'image, définition et utilisation du sliding window.

Dans la troisième partie, application du NMS sur les résultats obtenus.

## Références

* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [Stephan Dufner](http://www.duffner-net.de)