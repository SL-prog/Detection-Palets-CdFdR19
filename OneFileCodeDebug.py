import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import os


# region Fonctions
def ordoner_points(points_redressement):  # Renvoie les points ordones necessaire pour la suite
    # /!\ il peut y avoir une erreur si 2 points on la meme abcisse / ordonnee
    X = [points_redressement[i][0] for i in range(len(points_redressement))]
    orderX = sorted(X)
    Y = [points_redressement[i][1] for i in range(len(points_redressement))]
    orderY = sorted(Y)
    # On fait des listes ou on trie les x,y des points puis on selectionne les 2 points de gauche/haut/bas/droite
    gauche = [points_redressement[X.index(orderX[0])], points_redressement[X.index(orderX[1])]]  # points etant a gauche
    droite = [points_redressement[X.index(orderX[2])], points_redressement[X.index(orderX[3])]]
    haut = [points_redressement[Y.index(orderY[0])], points_redressement[Y.index(orderY[1])]]
    bas = [points_redressement[Y.index(orderY[2])], points_redressement[Y.index(orderY[3])]]

    def intersection(A, B):  # renvoie l'intersection entre 2 listes de points.
        if A[0] == B[0] or A[0] == B[1]:
            return A[0]
        else:
            return A[1]

    haut_gauche = intersection(haut, gauche)
    haut_droite = intersection(haut, droite)
    bas_gauche = intersection(bas, gauche)
    bas_droite = intersection(bas, droite)

    return [haut_gauche, haut_droite, bas_gauche, bas_droite]


def four_point_transform(image, pts, hauteur_mm, largeur_mm):  # renvoie l'image redresseee
    # ratio=ratio hauteur/largeur reele en mm
    print(hauteur_mm, largeur_mm)
    ratio = float(float(hauteur_mm) / float(largeur_mm))
    _, _, bas_gauche, bas_droite = pts
    largeur_pix = int(np.sqrt(((bas_droite[0] - bas_gauche[0]) ** 2) + ((bas_droite[1] - bas_gauche[1]) ** 2)))
    hauteur_pix = int(ratio * largeur_pix)

    distortion = np.array([[0, 0], [largeur_pix - 1, 0], [0, hauteur_pix - 1], [largeur_pix - 1, hauteur_pix - 1]],
                          dtype="float32")

    # calcul de la matrice de perspective
    M = cv2.getPerspectiveTransform(pts, distortion)  # pts en np.array (FLOAT32)

    # Matrice utilisee pour la fonction de redressmeent
    image_redressee = cv2.warpPerspective(image, M, (largeur_pix, hauteur_pix))
    return image_redressee


def conversion(data):  # Data be like [ [x0,y0,'color0'], [x1,y1,'color1'], [x2,y2,'color2'] ]
    return [conversion_largeur(data[0]), conversion_hauteur(data[1]), data[2]]


def conversion_largeur(x, Ax=1, Bx=0):  # A FAIRE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    global zero, largeur_mm, largeur_pix
    x = (x - zero[0]) * (largeur_mm / largeur_pix)
    x = Ax * x + Bx  # correction lineaire de la hauteur du palet (hauteur de noir en fonction de la distance) modelise par une droite
    return x


def conversion_hauteur(y, Ay=1, By=0):
    global zero, largeur_mm, largeur_pix
    y = (zero[1] - y) * (largeur_mm / largeur_pix)
    y = Ay * y + By  # correction lineaire de la "largeur" du palet ("largeur" de noir en fonction de la distance) modelise par une droite
    return y


# endregion fonctions


#######################


# region parametres du programme

# initialisation des variables pour eviter les erreurs dans l'ide (a cause du exec)
rayon_cercle_mm = None
# A MODIFIER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hauteur_mm, largeur_mm = None, None  # hauteur et largeur de la zone redressee en mm
decalage_redressement_robot = None  # distance entre le debut de l'image redressee et la base du robot //modifier et mettre centre du robot
resize_rate = None  # Taux de compression de l'image : image_init * resize_rate=image_finale

haut_gauche, haut_droite, bas_gauche, bas_droite = None, None, None, None
rayon_cercle_px = None  # rayon du cercle sur l'image redressee

seuils_detection = [None, None, None]  # B,G,R plus c'est eleve plus la hough gradient transform va detecter de cercles

aire_detection_contour = None  # Aire min, AireMax 0.75 coeff rectificateur, 0.8/1.2 : +- 20%

seuil_thresh = None  # seuil pour le thresh qui met 0 ou 255 pour chaque composante rgb


#import des parametres
try:
    file = open('parametres_programme.txt', 'r')
    constantes = file.read()
    file.close()
    exec(constantes)
except:
    raise Exception("Erreur dans l'import des parametres")

# endregion parametres

if not os.path.isdir("./out"):
    os.mkdir('./out')

file = open('pos_palets_mm.txt', 'w')
file.write('')
file.close()
###################


# region Aquisition de l'image
camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.resolution = (480, 360)  # (3280,2464)
camera.capture(rawCapture, format="bgr")
sleep(0.1)
image = rawCapture.array
# endregion aquisition image
image = cv2.imread("phototest2-09-05.jpg")
cv2.imwrite('./out/original.jpg', image)
if image is None:
    raise Exception("Erreur dans l'import. de l'image.    image==None")

##################


# region resize
largeur_init, hauteur_init, _ = np.shape(image)
image = cv2.resize(image, (int(hauteur_init * resize_rate), int(largeur_init * resize_rate)),
                   interpolation=cv2.INTER_AREA)
# endregion
cv2.imwrite('./out/resized.jpg', image)

###################


# region redressement
image = four_point_transform(image, np.array([haut_gauche, haut_droite, bas_gauche, bas_droite], np.float32),
                             hauteur_mm, largeur_mm)
# endregion redressement
cv2.imwrite('./out/redressement.jpg', image)

##################


# region thresh cad chaque composante couleur de chaque pixel devient 255 ou 0
thresh = cv2.threshold(image, seuil_thresh, 255, cv2.THRESH_BINARY)[1]

# cette ligne prend trop de temps, a ameliorer...
thresh[np.where((thresh == [255, 255, 255]).all(axis=2))] = [0, 0,
                                                             0]  # Convertis les pixels blancs en noirs pour eviter les mauvais countours
# endregion thresh
cv2.imwrite('./out/thresh.jpg', thresh)

##################


# region masks

# Traitement par couleurs
masks = []
masks_colors = [(255, 255, 0), (0, 255, 255), (
    0, 0, 255)]  # Le thresh ne donne pas du vert mais du jaune idem le bleu devient turquoise (le rouge est bon)
colors_string = ('b', 'g', 'r')

for i in range(3):
    mask = cv2.inRange(thresh, masks_colors[i], masks_colors[i])  # On conserve uniquement la couleur
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # On prend les countours des formes
    mask = np.zeros(np.shape(mask), np.uint8)  # Creation du masque final pour l'instant vierge

    for c in contours:
        # prrint(cv2.contourArea(c))
        if aire_detection_contour[0] < cv2.contourArea(c) < aire_detection_contour[
            1]:  # on ne garde que les contours d'aire convenable
            c = [c]
            cv2.drawContours(mask, c, -1, 255, -1)  # permet de remplir les trous
    masks.append(mask)
    cv2.imwrite('./out/mask' + colors_string[i] + '.jpg', mask)
# endregion masks


####################


# region Detection et Affichage des cercles
data = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
for i in range(3):
    circles = cv2.HoughCircles(masks[i], cv2.HOUGH_GRADIENT, 1, minDist=2 * rayon_cercle_px,
                               param2=seuils_detection[i], minRadius=int(rayon_cercle_px * 0.8),
                               maxRadius=int(
                                   rayon_cercle_px * 1.2))  # param2 : plus c'est grand moins ca detecte de cercles et plus c'est petit plus ca detecte de cercles
    # param1=264?
    if circles is not None:
        for j in circles[0]:
            data.append(
                [j[0], j[1],
                 colors_string[i]])  # on garde que les 2 premieres coord des cercles et on ajoute la couleur
        circles = np.uint16(np.around(circles))

        for j in circles[0, :]:  # dessin pour visualiser
            # draw the outer circle
            cv2.circle(image, (j[0], j[1]), j[2], colors[i], -1)
            # draw the center of the circle
            cv2.circle(image, (j[0], j[1]), 2, (0, 0, 0), 7)
# endregion
cv2.imwrite("./out/identification_cercle.jpg", image)

###################


# region conversion en mm des donnees
hauteur_pix, largeur_pix, _ = np.shape(image)
zero = (int(largeur_pix / 2), hauteur_pix)

for i in range(len(data)):
    data[i] = conversion(data[i])

# def dY(h):
#    return -int(54.6 + 0.0827 * h)
# def correction_lineaire(h):
#    return 0.1134 * h + 2.3975
# endregion


###########################


# region saving data
file = open('pos_palets_mm.txt', 'w')
outString = ''
for i in data:
    for j in i:
        outString += str(j) + ' '
    outString += '\n'

file.write(outString)
file.close()
# endregion saving data


################END
