import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


#rgb = cv2.imread('../IMG_7192.JPG')
rgb = cv2.imread('2018-06-08-100355_201x109_scrot.png')
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

I = (bp+gp-rp).astype('uint8')
I = cv2.equalizeHist(I).astype('int16')
#hist = cv2.calcHist([I],[0],None,[256],[0,256])
#plt.hist(I.ravel(),256,[0,256]); plt.show()
#plt.imshow(I, cmap = 'gray')

gray_levels = np.unique(I)
gray_levels_p =np.zeros(comb(gray_levels.size,2).astype('int16'))

#qi = gl graay_levels[j] = qj
for k,gl in enumerate(gray_levels):
    for j in range(k+1,gray_levels.size):
        sum_1 = np.log(np.abs(gl - gray_levels[j])+1)
        sum_2 = np.abs(np.where(I ==gl)[0].size*gl-np.where(I ==gray_levels[j])[0].size*gray_levels[j])/I.size
#        sum_3 = 
        gray_levels_p[k] = sum_1+sum_2
      
        
    


