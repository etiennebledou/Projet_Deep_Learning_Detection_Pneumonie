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


# Branche 01 : Configuration de l'environnement & Importation des bibliothèques

 
# Branche 02: Chargement du dataset provenant de Kaggle
- Le projet utilise le dataset : Chest X-Ray Pneumonia
- Images de radiographies thoraciques
- Deux classes : NORMAL et PNEUMONIA
- Structure : chest_xray/train/ val/ test/

 Problèmes identifiés
- Déséquilibre des classes (plus de PNEUMONIA que NORMAL)
- Ensemble de validation très faible
- Variabilité de qualité des images



# Branche 03 : Analyse Exploratoire (EDA)
- Distribution des classes
- Visualisation des images
- Analyse du déséquilibre
- Identification des biais potentiels
       - Résultat : un modèle naïf pourrait atteindre une bonne accuracy sans être utile.



# Branche 04 : Prétraitement des données
- Conversion en niveaux de gris
- Amélioration du contraste (CLAHE)
- Redimensionnement (224x224)
- Normalisation (ImageNet)
- Data augmentation : rotations/ flips/ zoom



# Branche 05 : GAN : Gestion du deséquilibre / reéquilibrage par génération synthétique
- Méthodes utilisées :
- Focal Loss
- Data Augmentation
- Génération d’images via GAN (DCGAN)



# Branche 06 : Modélisation

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


# Branche 07:  Explicabilité (XAI)

Utilisation de Grad-CAM pour :
- Visualiser les zones importantes de l’image
- Comprendre les décisions du modèle
- Vérifier la cohérence médicale


# Branche 08: Évaluation du Modèle

Métriques utilisées :
- Recall (prioritaire)
- Precision
- AUC
- Matrice de confusion
- Important : L’accuracy n’est pas utilisée seule car elle est trompeuse en cas de déséquilibre.


# Branche 09: Optimisation

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
- Ce projet a permis de développer un modèle de détection automatique de la pneumonie à partir de radiographies pulmonaires en combinant des techniques avancées de deep learning comm:
- CLAHE, normalisation 
- l’augmentation de données et le GAN 
- modèle hybride  optimisé avec fine-tuning et Optuna. 

Les résultats obtenus sont très satisfaisants :
- AUC ROC : 0.9736 
- Accuracy : 91.19% 
- Recall pneumonie élevé (0.97) garantissant peu de cas manqués
- Gain de +10.3 points sur le recall des cas normaux avec l’optimisation

Ce projet illustre comment une approche rigoureuse en data science peut contribuer à améliorer la détection précoce de maladies et soutenir les professionnels de santé.

