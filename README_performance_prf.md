# Performance Evaluation - Pseudo-Relevance Feedback (PRF)

Ce document explique comment utiliser le système d'évaluation de performance pour le Pseudo-Relevance Feedback.

## Description

Le script `src/performance/performance_prf.py` évalue les performances de récupération d'images avec Pseudo-Relevance Feedback (PRF) en utilisant :
- Différents modèles LLM
- Différents paramètres de l'algorithme Rocchio (alpha, beta, gamma)
- Les 1000 images sélectionnées de chaque dataset

## Structure des données

### Entrée
- Fichier d'entrée : `data/{dataset_name}/selected_1000.json`
- Contient les images à tester avec leurs métadonnées (filename, caption, baseline_rank, etc.)

### Sortie
- Répertoire de sortie : `report/performance_prf/{dataset_name}/`
- Fichiers générés :
  - `performance_{llm_model}_a{alpha}_b{beta}_g{gamma}.json` : Résultats détaillés pour chaque configuration
  - `evaluation_summary.json` : Résumé global de toutes les évaluations

## Paramètres de Rocchio

L'algorithme Rocchio modifie la requête initiale selon la formule :
```
Q_modifiée = α * Q_originale + β * centroïde_docs_pertinents - γ * centroïde_docs_non_pertinents
```

- **alpha (α)** : Poids de la requête originale (défaut: 1.0)
- **beta (β)** : Poids des documents pertinents (défaut: 0.75)
- **gamma (γ)** : Poids des documents non pertinents (défaut: 0.0)

Configurations testées par défaut :
1. `(1.0, 0.5, 0.0)` - Feedback conservateur
2. `(1.0, 0.75, 0.0)` - Feedback modéré (défaut)
3. `(1.0, 1.0, 0.0)` - Feedback agressif
4. `(0.8, 0.8, 0.0)` - Équilibré
5. `(1.5, 0.5, 0.0)` - Emphase sur la requête originale

## Modèles LLM testés

Par défaut, les modèles suivants sont évalués :
- `mistral:7b`
- `gpt-oss:20b`
- `gemma3:4b`
- `gemma3:27b`

## Utilisation

### Évaluation complète (tous LLM, tous paramètres Rocchio)

```bash
python src/performance/performance_prf.py --dataset COCO
python src/performance/performance_prf.py --dataset Flickr
python src/performance/performance_prf.py --dataset VizWiz
```

### Évaluation d'une configuration spécifique

```bash
# Évaluer un LLM spécifique avec des paramètres Rocchio personnalisés
python src/performance/performance_prf.py --dataset COCO --llm-model "mistral:7b" --alpha 1.0 --beta 0.75 --gamma 0.0
```

### Paramètres optionnels

- `--dataset` : Dataset à évaluer (COCO, Flickr, VizWiz) - **REQUIS**
- `--k` : Nombre de résultats à récupérer (défaut: 10)
- `--llm-model` : Modèle LLM spécifique à évaluer
- `--alpha` : Paramètre alpha de Rocchio (requiert --beta)
- `--beta` : Paramètre beta de Rocchio (requiert --alpha)
- `--gamma` : Paramètre gamma de Rocchio (défaut: 0.0)

## Structure du fichier de résultats

Chaque fichier JSON contient :

```json
{
  "metadata": {
    "dataset": "COCO",
    "llm_model": "mistral:7b",
    "rocchio_params": {
      "alpha": 1.0,
      "beta": 0.75,
      "gamma": 0.0
    },
    "k": 10,
    "num_images": 1000,
    "evaluation_date": "2025-11-06T...",
    "elapsed_time_seconds": 3600.5
  },
  "image_results": [
    {
      "image_id": 12345,
      "filename": "000000012345.jpg",
      "caption": "A cat sitting on a table",
      "baseline_rank": 17,
      "baseline_similarity": 0.8234,
      "llm_evaluation": {
        "relevant_indices": [1, 3, 5],
        "num_relevant": 3,
        "reasoning": "Images 1, 3, and 5 show cats in similar poses"
      },
      "prf_rank": 8,
      "prf_similarity": 0.8567,
      "rank_improvement": 9,
      "baseline_top_k": [...],
      "prf_top_k": [...]
    },
    ...
  ],
  "statistics": {
    "total_images": 1000,
    "valid_images": 995,
    "error_images": 5,
    "improvements": {
      "count": 450,
      "percentage": 45.2,
      "avg_improvement": 12.3
    },
    "degradations": {
      "count": 300,
      "percentage": 30.2,
      "avg_degradation": -8.5
    },
    "no_changes": {
      "count": 150,
      "percentage": 15.1
    },
    "llm_found_relevant": {
      "count": 800,
      "percentage": 80.4,
      "avg_relevant_per_image": 2.5
    }
  }
}
```

## Métriques calculées

Pour chaque image évaluée :
- **baseline_rank** : Rang de l'image cible dans les résultats baseline (CLIP seul)
- **prf_rank** : Rang de l'image cible après application du PRF
- **rank_improvement** : Différence de rang (positif = amélioration)
- **llm_evaluation** : Résultats de l'évaluation LLM (images pertinentes identifiées)

Statistiques globales :
- Nombre et pourcentage d'améliorations
- Nombre et pourcentage de dégradations
- Amélioration moyenne de rang
- Taux de succès du LLM (% d'images où le LLM a trouvé des documents pertinents)

## Flux de traitement

Pour chaque image dans `selected_1000.json` :

1. **Récupération baseline** : Utilisation de CLIP pour récupérer les top-k images similaires
2. **Génération de descriptions BLIP** : Création de captions pour les images récupérées
3. **Évaluation LLM** : Le LLM identifie les images pertinentes parmi les top-k
4. **Application PRF** : Si des images pertinentes sont trouvées, application de l'algorithme Rocchio
5. **Ré-récupération** : Nouvelle recherche avec la requête modifiée
6. **Tracking** : Suivi du rang de l'image cible avant/après PRF
7. **Sauvegarde** : Enregistrement des résultats

## Notes importantes

- Les résultats sont sauvegardés progressivement (tous les 50 images) pour éviter la perte de données
- L'évaluation complète peut prendre plusieurs heures selon le nombre de configurations
- Les paramètres Rocchio peuvent maintenant être personnalisés via le constructeur de `ImageRetriever`
- Le fichier `evaluation_summary.json` fournit une vue d'ensemble comparative de toutes les configurations

## Exemples de commandes

```bash
# Évaluation complète du dataset COCO
python src/performance/performance_prf.py --dataset COCO

# Évaluation rapide avec un seul LLM et des paramètres spécifiques
python src/performance/performance_prf.py --dataset Flickr --llm-model "gemma3:4b" --alpha 1.0 --beta 0.5 --gamma 0.0

# Évaluation avec k=20 résultats
python src/performance/performance_prf.py --dataset VizWiz --k 20
```
