# Lab 5: Feature Store & Model Packaging

## 🎯 Services AWS Utilisés dans ce Lab
- ✅ **SageMaker Feature Store** (Online + Offline storage)
- ✅ **SageMaker Model Registry** (Versioning & approval)
- ⚙️ Model Packaging (model.tar.gz, inference scripts)

## 🎯 Objectifs d'Apprentissage

À la fin de ce lab, vous serez capable de :

1. **Créer un Feature Store** et ingérer des données (Online + Offline)
2. **Récupérer des features** pour training et inference
3. **Packager un modèle** au format `model.tar.gz` pour SageMaker
4. **Créer des scripts d'inférence** personnalisés (`inference.py`)
5. **Gérer les dépendances** avec `requirements.txt`
6. **Enregistrer le modèle** dans le Model Registry avec versioning
7. **Approuver des modèles** pour déploiement production

## 📚 Concepts Couverts

### 1. SageMaker Feature Store
- Création de Feature Groups (Online + Offline)
- Ingestion de données temps réel
- Récupération de features pour training
- Time-travel queries (point-in-time)
- Feature discovery et réutilisation

### 2. Structure du Package Modèle
```
model.tar.gz
├── model.pkl (ou model.pth, saved_model.pb, etc.)
├── inference.py (optionnel)
├── requirements.txt (optionnel)
└── code/ (optionnel)
    ├── preprocessing.py
    └── utils.py
```

### 3. Scripts d'Inférence
- `model_fn()`: Chargement du modèle
- `input_fn()`: Parsing des requêtes
- `predict_fn()`: Prédiction
- `output_fn()`: Formatage des réponses

### 4. Model Registry
- Enregistrement de modèles avec métadonnées
- Versioning automatique
- Workflow d'approbation
- Lineage tracking

## 🧪 Exercices Pratiques

### Exercice 1: Créer un Feature Store
Créer Feature Groups pour customer features (Online + Offline)

### Exercice 2: Ingérer et Récupérer des Features
Ingérer des données et requêter le Feature Store

### Exercice 3: Package Modèle avec Features
Packager un modèle avec dépendances Feature Store

### Exercice 4: Script d'Inférence avec Feature Store
Créer `inference.py` qui utilise le Feature Store

### Exercice 5: Enregistrer dans Model Registry
Enregistrer le modèle avec métadonnées et approval workflow

## ⏱️ Durée Estimée
- Exercices: 60 minutes
- Solution review: 30 minutes
- **Total: 90 minutes**

## 📋 Prérequis

- Complétion des Labs 1-4 (ML de base)
- Compréhension de Python et pip
- Connaissance de base de Docker (pour exercice 4)
- Modèle entraîné disponible

## 🚀 Getting Started

1. Ouvrir `exercises/packaging_exercise.ipynb`
2. Suivre les instructions étape par étape
3. Comparer avec `solutions/packaging_solution.ipynb`

## 📊 Ce que Vous Allez Construire

- ✅ Package de modèle production-ready
- ✅ Script d'inférence optimisé
- ✅ Container Docker personnalisé
- ✅ Tests automatisés de packaging
- ✅ Documentation du modèle

## 🔗 Ressources

- [SageMaker Model Packaging](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html)
- [Custom Inference Code](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html)
- [Docker Containers for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
