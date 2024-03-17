# Le stockage de simulation

Comment est stocké une simulations pré-simulée.

extension `.sim`

## Header

2 x 16 bits (2 x 2 octets) pour la taille du canvas.
- 16 bits : uint16 taille X de la sim
- 16 bits : uint16 taille Y de la sim

***TODO : add other metadata***

## Data

Chaque *frame* de la simulation contient les données de chaque case.

$offset$ est le décalage jusqu'à la première case.
$X$ la taille X de la sim.
Pour obtenir la position de la case aux coordonnées $x, y$ :

$$
case(x, y) = offset + x + y \times X
$$

Si le bit est `0`, la case est vide.
Si le bit est `1`, il y aura d'autres données derrière (case occupée).

### Données Entité

Une case occupée commencera par un `1` pour indiquer son occupation.
