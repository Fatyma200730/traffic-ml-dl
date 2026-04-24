# Système de Détection et Prédiction du Trafic Web

**Projet de Fin d'Études — 2025/2026**
**CRISP-DM — Phase 1 : Business Understanding**

---

## 1. Introduction

Aujourd'hui, presque tous les services passent par le web : sites e-commerce, applications, plateformes éducatives, etc. Chaque visite, chaque clic génère du trafic sur un serveur.

Surveiller ce trafic est essentiel. Sans surveillance, un pic soudain peut faire tomber un service, une attaque peut passer inaperçue, et les équipes techniques réagissent toujours trop tard.

Ce projet propose un système intelligent capable d'analyser le trafic web, de détecter les comportements anormaux et de prédire son évolution.

---

## 2. Problématique

**Comment détecter automatiquement les anomalies dans le trafic web et anticiper son évolution ?**

Sans outil adapté, une organisation fait face à plusieurs risques :

- Surcharge du serveur → panne et indisponibilité du service
- Attaques non détectées (ex : DDoS, bots malveillants)
- Baisse de performance → mauvaise expérience utilisateur
- Décisions basées sur des estimations et non sur des données réelles

---

## 3. Objectifs

**Objectif principal** : détecter les anomalies du trafic web et prédire son évolution future.

**Objectifs secondaires** :
- Améliorer la sécurité en identifiant les comportements suspects
- Optimiser les performances en anticipant les pics de charge
- Aider à la prise de décision grâce à des visualisations claires

---

## 4. Choix des données ⭐

Ce projet repose sur des **données temporelles (time series)** — c'est-à-dire des mesures du trafic enregistrées à intervalles réguliers dans le temps.

**Caractéristiques attendues du dataset :**

| Colonne | Description |
|---|---|
| `timestamp` | Date et heure de la mesure |
| `visitors` | Nombre de visiteurs |
| `pageviews` | Nombre de pages vues |
| `sessions` | Nombre de sessions ou requêtes |

**Pourquoi ce type de données ?**
- Adapté à l'analyse temporelle (patterns quotidiens, hebdomadaires)
- Compatible avec les algorithmes de Machine Learning et Deep Learning
- Permet de détecter des anomalies : pics, chutes soudaines, comportements inhabituels

**Sources envisagées :**
- [Wikipedia Web Traffic Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) — Kaggle
- Google Trends — données réelles d'évolution de recherches web

---

## 5. Importance métier

Ce système apporte une valeur concrète pour toute organisation gérant un service en ligne :

- **Disponibilité** : détecter un problème avant qu'il impacte les utilisateurs
- **Sécurité** : repérer les attaques ou comportements anormaux rapidement
- **Optimisation** : ajuster les ressources serveur selon les prévisions de trafic
- **Décision** : fournir aux équipes des données fiables plutôt que des suppositions

---

## 6. Conclusion

Cette première phase CRISP-DM est la base de tout le projet. Elle permet de répondre à une question simple avant de commencer à coder : **quel problème on résout, avec quelles données, et pourquoi ?**

Le choix des données est particulièrement important ici. Des données temporelles bien structurées conditionneront directement la qualité des modèles de détection et de prédiction dans les phases suivantes.

---

*Étudiante : [Prénom Nom] — EST Fkih Ben Salah, Université Sultan Moulay Slimane*
*Encadrant : [Nom de l'encadrant]*
