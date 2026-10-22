# CREDITIA — Modèle de scoring du risque de crédit en microfinance

**Un système d'aide à l'évaluation du risque de crédit combinant scoring expert, analyse de la capacité de remboursement et apprentissage automatique.**

## Présentation du projet

CREDITIA est un projet de Data Science conçu pour explorer l'utilisation de l'analyse de données et du Machine Learning dans l'évaluation du risque de crédit en microfinance.

Le modèle combine une approche de scoring expert, une analyse de la capacité de remboursement, des contrôles d'éligibilité et un modèle prédictif basé sur **XGBoost**.

L'objectif est d'aider les institutions de microfinance à mieux analyser les dossiers de demande de crédit, à estimer les capacités de remboursement et à identifier les dossiers nécessitant une analyse complémentaire.

Le projet porte principalement sur cinq pays d'Afrique de l'Ouest : le Bénin, le Burkina Faso, le Mali, le Sénégal et le Togo.

**Statut :** Prototype de recherche et développement.

## Objectifs

- Évaluer la capacité de remboursement des demandeurs de microcrédit.
- Combiner les critères financiers et qualitatifs dans un score sur 100 points.
- Estimer la probabilité de défaut à l'aide de XGBoost.
- Proposer une classification du risque de crédit.
- Intégrer des contrôles d'éligibilité et des règles de conformité.
- Faciliter l'analyse des dossiers grâce à une éventuelle interface interactive avec Streamlit.
- Contribuer à une prise de décision plus structurée et documentée.

## Fonctionnalités principales

### 1. Scoring expert

Le score expert est structuré autour de deux composantes :

**Score quantitatif — 60 points**
- Capacité de remboursement : 25 points
- Régularité des revenus et remboursements : 10 points
- Endettement existant : 10 points
- Historique interne et informations de crédit : 10 points
- Épargne disponible : 5 points

**Score qualitatif — 40 points**
- Continuité de l'activité ou de l'emploi
- Cohérence entre le montant, l'objet et la durée du crédit
- Discipline de gestion
- Résilience et diversification des revenus
- Stabilité de l'employeur ou du marché
- Stabilité des coordonnées et du lieu de résidence
- Garanties ou soutien disponible

Le score expert final est exprimé sur 100 points.

### 2. Analyse de la capacité de remboursement

Le modèle distingue trois profils de revenus :

- **Salariés et pensionnés :** revenus nets vérifiés, charges essentielles, endettement et stabilité des revenus.
- **Travailleurs indépendants :** recettes, coûts variables, charges professionnelles et dépenses du ménage.
- **Activités saisonnières ou revenus mixtes :** recettes par cycle, coûts d'activité, charges du ménage et durée des cycles.

L'analyse vise à estimer une mensualité soutenable et un montant de crédit compatible avec la capacité de remboursement du demandeur.

### 3. Modèle de Machine Learning

CREDITIA utilise **XGBoost** pour expérimenter la prédiction du risque de défaut de remboursement.

La variable cible est `default_90d` :

- `1` : le client atteint au moins 90 jours de retard de paiement pendant la période d'observation.
- `0` : la période d'observation est complète et le retard maximal reste inférieur à 90 jours.

Les dossiers dont la période d'observation est incomplète ne doivent pas être automatiquement considérés comme non défaillants.

Le pipeline de préparation des données peut inclure :
- L'imputation
