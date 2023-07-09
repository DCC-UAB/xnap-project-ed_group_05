[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122280&assignment_repo_type=AssignmentRepo)
# COLOURING GRAYSCALE PHOTOS
Aquest repositori conté diversos codis per a coloregar imatges en blanc i negre utilitzant diferents tècniques i models d'aprenentatge automàtic. A continuació, s'explica breument cada un dels codis disponibles:

## Beta
Aquest codi implementa un model de xarxa neuronal per a la coloració d'imatges en blanc i negre. Carrega un conjunt d'imatges en blanc i negre i divideix les dades en conjunts d'entrenament i de prova. El model utilitza diverses capes de convolució i submostreig per a l'extracció de característiques i l'augment de la resolució. S'utilitza l'optimitzador RMSprop i la funció de pèrdua MSE per a l'entrenament del model.
Una vegada entrenat el model, es fa una avaluació de la pèrdua utilitzant les imatges de prova. A continuació, s'utilitza el model per a predir els colors de les imatges de prova i es generen imatges en color a partir d'aquestes prediccions.
Per utilitzar aquest codi, cal proporcionar un conjunt d'imatges en la carpeta especificada. S'han de definir els paràmetres de l'optimitzador i de l'entrenament segons les necessitats. El codi guarda els pesos del model, genera gràfics de la pèrdua durant l'entrenament i guarda les imatges en color resultants.
Es recomana ajustar els paràmetres i experimentar amb diferents tècniques d'augmentació de dades per a obtenir els millors resultats.


Original      vs         Predita

<img width="206" alt="image" src="https://github.com/DCC-UAB/xnap-project-ed_group_05/assets/102381651/fdd11002-d278-44e3-b075-1cc26396a27a">

## Starting
Com a punt de partida del projecte teniem el GitHub que inclou el dataset ‘face_images’ a més d’una carpeta amb dos codis el Versió Alpha i el Versió Beta. En aquest cas no hem volgut aprofitar cap d’aquestes facilitats ja que el dataset té imatges amb les mateixes tonalitats i per tant el model no hauria d’aprendre massa ja que hi ha poca complexitat i, en quan als codis, en la primera entrega ja vam comprovar que no funcionaven corrrectament i per tant no els anàvem a reaprofitar com a tal.

## 255
En aquest cas el plantejament és diferent i el que volem és fer una normalització concreta per les dades d’entrada (Xtrain) i les dades de sortida (Ytest).
Primerament, igual que en el cas anterior, és important fer primer la transformació RgbtoLab i després ja podem començar a aplicar la normalització.
En aquest cas les dades d’entrada es pot normalitzar simplement dividint per 255.0. Així aconseguim escalar els valors dels píxels en el rang [0, 1], ja que originalment es troben en el rang [0, 255].
Per altre banda, les dades de sortida es fa una normalització diferent, els valors d’aquesta capa es divideix per 128.0 i això ens permet ajustar els valors en un rang [-1, 1]. El motiu és que en l’espai de color LAB aquests valors es centren al voltant de zero amb un rang típic de [-128, 128]. Dividint per 128.0 ens proporciona una manera convenient d’escalar valors de rang [-1, 1] que pot facilitar l’entrenament i la convergència del model.

## Desviació típica
Aquest codi la diferència principal que té és que calcula la mean (mitjana) i la sd (desviació típica) de les dades per tal d’obtenir els valors de les imatges entre [-1, 1], però com treballem amb la desviació típica poden ballar una mica més i podrien arribar a valors dins l’interval [-3, 3].
D’aquesta manera es fa la normalització, però en aquest cas és molt important mantenir un ordre coherent i per això necessitem fer primerament la conversió RgbtoLab i després ja la normalització explicada.


## Com utilitzar els codis
Cada codi està contingut en un fitxer separat amb l'extensió .py. Per utilitzar-los, segueix aquests passos:
1. Assegura't de tenir les imatges o els gifs que vulguis utilitzar en el directori del codi.
2. Actualitza la ruta del directori dins del codi.
3. Executa el fitxer corresponent al codi que vulguis utilitzar.
4. Espera que el codi completi el procés de colorejat.
5. Comprova els resultats, els quals es guardaran en el directori del codi.
## Requeriments
Per executar els codis, són necessàries les següents dependències:

Python 

Biblioteques de Python:
  - Numpy
  - Keras
  - TensorFlow
  - Skimage
  - Matplotlib
  - Random
  - Os
  - PIL
  - Imageio
  - WandB
## Referències
[Coloring black and white images with neural networks](https://github.com/emilwallner/Coloring-greyscale-images)
[WandB](https://wandb.ai/team_ev/my-awesome-project?workspace=user-videelia)
## Contribuïdores
Àfrica Gamero López (1606033@uab.cat), Èlia Vinyes Devesa (1606019@uab.cat), Marina Sulé Carrasco (1527602@uab.cat)

Xarxes Neuronals i Aprenentatge Profund

Grau d'Enginyeria de Dades, 

UAB, 2023
