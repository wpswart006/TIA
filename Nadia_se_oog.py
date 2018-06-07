import cv2
import numpy as np
import matplotlib.pyplot as plt

rgb = cv2.imread('../IMG_7187.JPG')
rgb = cv2.resize(rgb,(640,480))

r = rgb[:,:,2]
g = rgb[:,:,1]
b = rgb[:,:,0]

nR = np.mean(r) + np.std(r)/2
nG = np.mean(g) + np.std(g)/2
nB = np.mean(b) + np.std(b)/2

mu = np.min([nR,nG,nB])

a = 2 + np.sqrt(3)
_b = 7-4*np.sqrt(3)

rp = 128 * (1 + (1-np.power(_b,r/(1.5*mu))/(a*np.power(_b,r/(1.5*mu))+1)))
gp = 128 * (1 + (1-np.power(_b,g/(1.5*mu))/(a*np.power(_b,g/(1.5*mu))+1)))
bp = 128 * (1 + (1-np.power(_b,b/(1.5*mu))/(a*np.power(_b,b/(1.5*mu))+1)))

I = bp+gp-rp
plt.imshow(I)