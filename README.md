# Detection-Palets-CdFdR19

RaspberryPI

## Installation environnement virtuel
http://raspberrypi-aa.github.io/session4/venv.html

```
sudo pip install virtualenv
cd envopencvcoupe19
source bin/activate
```

## Installation OpenCV + PiCamera

```
sudo apt-get update
sudo apt-get upgrade
sudo python3 get-pip.py
sudo pip install opencv-contrib-python
sudo pip install PiCamera
```

###### Note : ne pas mettre " sudo " si travail dans environnement virtuel

## Tester

```
python
>>> import cv2
>>> cv2.__version__
>>> from picamera import PiCamera
```

###### Note : Si Python ne reconnait pas cv2, essayer " python2 " ou " python3 " (le programme fonctionne sous Python 2.7 et Python 3).
