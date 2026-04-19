# Projet_Deep_Learning_Detection_Pneumonie
Projet de contexte medical portant sur la detection de la Pneumonie par Deep Learning

DESCRIPTION DU PROJET

Ce projet vise à développer un modèle de Deep Learning capable de détecter la pneumonie à partir d’images de radiographies pulmonaires.
L’objectif principal est de construire un système fiable d’aide au diagnostic médical, en mettant l’accent sur la détection des cas positifs (Recall) afin de minimiser les risques de faux négatifs.


 Objectifs
- Détecter automatiquement la pneumonie à partir d’images médicales
- Gérer le déséquilibre des données
- Maximiser le Recall (sensibilité) pour les cas de pneumonie
- Interpréter les décisions du modèle (Explainable AI)
- Comparer plusieurs architectures de Deep Learning


# BRANCHE 01 : CONFIGURATION DE L'ENVIRONNEMENT & IMPORTATION DES BIBLIOTHEQUES

 
# BRANCHE 02: CHARGEMENT DU DATASET VIA KAGGLE API
- Le projet utilise le dataset : Chest X-Ray Pneumonia
- Images de radiographies thoraciques
- Deux classes : NORMAL et PNEUMONIA
- Structure : chest_xray/train/ val/ test/

 Problèmes identifiés
- Déséquilibre des classes (plus de PNEUMONIA que NORMAL)
- Ensemble de validation très faible
- Variabilité de qualité des images



# BRANCHE 03 : ANALYSE EXPLORATOIRE (EDA)
- Distribution des classes
- Visualisation des images
- Analyse du déséquilibre
- Identification des biais potentiels
       - Résultat : un modèle naïf pourrait atteindre une bonne accuracy sans être utile.



# BRANCHE 04 : PRETRAITEMENT DES DONNEES
- Conversion en niveaux de gris
- Amélioration du contraste (CLAHE)
- Redimensionnement (224x224)
- Normalisation (ImageNet)
- Data augmentation : rotations/ flips/ zoom



# BRANCHE 05 : GAN : REEQUILIBRAGE PAR GENERATION SYNTHETIQUE/ GESTION DU DESEQUILIBRE
  
  Méthodes utilisées :
- Focal Loss
- Data Augmentation
- Génération d’images via GAN (DCGAN)



# BRANCHE 06 : MODELISATION

Modèles utilisés
- CNN Baseline
- ResNet18 (Transfer Learning)
- DenseNet121
- EfficientNet
- CNN + Vision Transformer

Méthodes d’entraînement
- Transfer Learning
- Fine-Tuning
- Early Stopping (basé sur Recall)
- Learning Rate Scheduler



# BRANCHE 07: ÉVALUATION

Métriques utilisées :
- Recall (prioritaire)
- Precision
- AUC
- Matrice de confusion
 - Important : L’accuracy n’est pas utilisée seule car elle est trompeuse en cas de déséquilibre.



# BRANCHE 08:  EXPLICABILITE (XAI)

Utilisation de Grad-CAM pour :
- Visualiser les zones importantes de l’image
- Comprendre les décisions du modèle
- Vérifier la cohérence médicale


# BRANCHE 09: OPTIMISATION 

- Optimisation des hyperparamètres avec Optuna
- Ajustement du seuil de classification
- Résultats
- Amélioration significative du Recall
- Meilleure détection des cas de pneumonie
- Modèle robuste malgré le déséquilibre


#  Limites
- Dataset biaisé (population spécifique)
- Peu de données
- Pas de validation externe
- Risque d’overfitting

 
 Améliorations possibles
- Validation croisée (K-Fold)
- Utilisation d’un dataset externe
- Calibration des probabilités
- Déploiement en application web (Streamlit / API)


 Technologies utilisées
- Python
- PyTorch / TensorFlow
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn


 
  Structure du projet
├── data/
├── notebooks/
│   └── Pneumonie.ipynb
├── models/
├── results/
├── README.md


 
  Installation
- git clone https://github.com/etiennebledou/Projet_Detection_Pneumonie_par_Deep_Learning.git
- cd Projet_Detection_Pneumonie_par_Deep_Learning
- pip install -r requirements.txt


 Utilisation
- jupyter notebook Pneumonie.ipynb / directement sur Google colab

 
  Equipe
- Etienne Bledou
- Samba Diakho
- Hanane Derbak
- Youva Hamani


 Licence
- Projet académique – usage éducatif


 Remarque
- Ce projet illustre l’application du Deep Learning dans le domaine médical, avec une attention particulière portée à la fiabilité, à l’interprétabilité et aux enjeux éthiques.
