# PhaseLab

Analyse de phase, de retard et de réponse en fréquence pour comparer des prises
micro. Le signal de test est généré par PhaseLab, joué et enregistré dans
Reaper, puis les prises sont analysées ici.

## Installation

Il faut Python 3.10 ou plus récent (python.org).

- **Windows** : double-clic sur `Lancer PhaseLab (Windows).bat`
- **Mac** : clic droit sur `Lancer PhaseLab (Mac).command` › Ouvrir
  (la première fois, `chmod +x` peut être nécessaire)

Le premier lancement installe les dépendances (une minute). Les suivants sont
immédiats. Manuellement : `pip install -r requirements.txt` puis
`python phaselab.py`.

## Essayer tout de suite

Le dossier `exemple/` contient un sweep et cinq prises simulées (0° à 180°,
avec retards, polarités et rotations de phase connus). Glisse-les tous dans la
fenêtre de PhaseLab.

## Utilisation en 3 étapes

1. **Générer** un signal (sweep logarithmique recommandé) à la fréquence
   d'échantillonnage de ton projet Reaper. Garde le `.wav` et le
   `.phaselab.json` ensemble.
2. **Dans Reaper** : signal posé à 0:00 vers l'enceinte, une piste par micro,
   enregistrement depuis 0:00, sans effet. Une prise par angle de micro, nommée
   avec l'angle (`snare_45deg.wav`). Exporte avec *Render › Stems*, bornes
   *Entire project*, sans normalisation ni dither.
3. **Importer** les prises (bouton ou glisser-déposer).

Le menu *Aide › Guide* reprend tout ça dans l'appli.

## Ce que montre chaque onglet

| Onglet | Contenu |
|---|---|
| Réponse | Niveau, phase, retard de groupe de chaque prise. Survol = valeurs exactes. |
| Temps | Réponse impulsionnelle, enveloppe ETC, et prise comparée à l'original (aligné, ou filtré par la réponse mesurée : le résidu ne contient alors que bruit + distorsion). |
| Différences | Δ niveau et Δ phase par rapport à la référence, sommation avant/après correction, tableau des réglages. |
| Rotation | Δ phase, Δ niveau et Δ temps en fonction de l'angle, carte fréquence × angle. |

Le **tableau des réglages** donne pour chaque prise : retard (ms, échantillons,
cm), polarité, rotation de phase constante restante, gain, écart résiduel (ce
que retard + rotation n'expliquent pas) et la première fréquence d'annulation
si on somme les deux micros tels quels.

## Méthodologie (équivalent des outils MATLAB de labo)

- **Excitation** : sweep exponentiel synchronisé (Farina 2000, Novak 2015).
- **Déconvolution** : division spectrale régularisée (inversion de Kirkeby).
  La distorsion harmonique tombe en temps négatif et est écartée.
- **Repère temporel absolu** : temps 0 = début du fichier du signal, ou canal
  loopback (méthode à double canal) pour éliminer latence et coloration de
  l'interface.
- **Pic sous-échantillon** : interpolation sinc.
- **Fenêtrage** de la réponse impulsionnelle (demi-Hann), réglable.
- **Lissage complexe** par fraction d'octave après retrait du retard propre ;
  retard de groupe calculé raie par raie puis moyenné par l'énergie.
- **Retard + rotation de phase** estimés conjointement (corrélation croisée à
  pondération de phase et 1/f, enveloppe analytique, interpolation) dans la
  bande choisie.

Validé sur des prises simulées : retards retrouvés à 0,001 ms près, polarité,
rotation (±0,1°) et gain (±0,01 dB) exacts.

## Exports

Courbes de réponse et de différence (CSV, séparateur `;`), tableau des
réglages, courbes de correction EQ (format texte REW), données de rotation,
réponses impulsionnelles (WAV 32 bits flottant, temps 0 conservé : utilisables
directement en convolution dans Reaper) et image de l'onglet.

## Fichiers

- `phaselab.py` : interface
- `dsp.py` : moteur d'analyse (utilisable seul depuis Python)
- `style.py` : thème
