# PyRPS

Une compétition évolutive à base d'IA, de pierres, de feuilles, de ciseaux et d'autres mécanismes intéressants pour déterminer quel camp est le meilleur.

# Installation

Les librairies nécessaires à l'exécution de la simulation sont `pygame`, `matplotlib` et `NumPy` :

```cmd
> pip install pygame matplotlib numpy
```

# Simulation

Pour lancer la simulation :

```cmd
> python example.py
```

Cela ouvrira une fenêtre graphique pygame, contenant chaque camp représenté par des points de sa couleur (rouge, vert et bleu). Lorsqu'il ne reste plus qu'une seule couleur, fermer la fenêtre pygame affichera un graphique représentant l'évolution des trois population au cours du temps.

# Résolution de problèmes

## La fenêtre avec le graphique ne s'affiche pas

Si après la fin de la simulation, la fenêtre avec le graphique ne s'ouvre pas et que l'erreur suivante est affichée :

```cmd
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

Il faut installer la librairie `PyQt5` :

```cmd
> pip install PyQt5
```