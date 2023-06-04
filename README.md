[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122280&assignment_repo_type=AssignmentRepo)
# COLOURING GRAYSCALE PHOTOS
Aquest repositori conté diversos codis per a coloregar imatges en blanc i negre utilitzant diferents tècniques i models d'aprenentatge automàtic. A continuació, s'explica breument cada un dels codis disponibles:

## Alpha
Aquest codi implementa un model de xarxa neuronal per a la coloració d'imatges en blanc i negre. Carrega una imatge, la converteix a l'espai de colors Lab i utilitza aquesta imatge per a entrenar el model. El model té diverses capes de convolució i submostreig per a l'extracció de característiques i l'augment de la resolució. Finalment, el model és entrenat utilitzant les dades d'entrada i sortida i es genera una imatge en color a partir de la imatge en blanc i negre.
Per utilitzar aquest codi, simplement carrega una imatge en blanc i negre, ajusta els paràmetres d'entrenament (com ara el nombre d'èpoques) i executa el codi per a generar la imatge en color. Pots ajustar el model i experimentar amb diferents paràmetres per a obtenir resultats òptims.
## Beta
Aquest codi implementa un model de xarxa neuronal per a la coloració d'imatges en blanc i negre. Carrega un conjunt d'imatges en blanc i negre i divideix les dades en conjunts d'entrenament i de prova. El model utilitza diverses capes de convolució i submostreig per a l'extracció de característiques i l'augment de la resolució. S'utilitza l'optimitzador RMSprop i la funció de pèrdua MSE per a l'entrenament del model.
Durant l'entrenament, les imatges d'entrenament es transformen utilitzant tècniques d'augmentació de dades com ara la rotació, el volteig horitzontal, el desplaçament i el zoom. El model és entrenat utilitzant aquestes imatges transformades i les seves corresponents imatges en color com a dades d'entrada i sortida.
Una vegada entrenat el model, es fa una avaluació de la pèrdua utilitzant les imatges de prova. A continuació, s'utilitza el model per a predir els colors de les imatges de prova i es generen imatges en color a partir d'aquestes prediccions.
Per utilitzar aquest codi, cal proporcionar un conjunt d'imatges en blanc i negre en la carpeta especificada. S'han de definir els paràmetres de l'optimitzador i de l'entrenament segons les necessitats. El codi guarda els pesos del model, genera gràfics de la pèrdua durant l'entrenament i guarda les imatges en color resultants.
Es recomana ajustar els paràmetres i experimentar amb diferents tècniques d'augmentació de dades per a obtenir els millors resultats.
## Gifs
El codi "gif" és una extensió del codi "alpha" que permet colorejar gifs en blanc i negre. Aquest codi utilitza la mateixa aproximació basada en l'entrenament d'un model d'aprenentatge automàtic, però en aquest cas es processa cada fotograma del gif per separat i després es recomponen per a generar el gif final en color.
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
## Referències
[Coloring black and white images with neural networks](https://github.com/emilwallner/Coloring-greyscale-images)
## Contribuïdores
Àfrica Gamero López (1606033@uab.cat), Èlia Vinyes Devesa (1606019@uab.cat), Marina Sulé Carrasco (1527602@uab.cat)

Xarxes Neuronals i Aprenentatge Profund

Grau d'Enginyeria de Dades, 

UAB, 2023
