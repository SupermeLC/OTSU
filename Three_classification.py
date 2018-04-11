from PIL import Image
from pylab import *
from skimage import filters
import numpy as np

def OTSU(im_gray):
    g_min = 9999999
    g_max = 0
    best_th = 0
    for threshold1 in range(0,256,5):
        for threshold2 in range(0,256,5):
            logic2 = im_gray > threshold2
            logic1 = (im_gray <= threshold2) & (im_gray > threshold1)
            logic0 = im_gray <= threshold1
            fore_pix = np.sum(logic2)
            mid_pix = np.sum(logic1)
            back_pix = np.sum(logic0)
            
            if fore_pix == 0: #END
                break
            if mid_pix == 0: #CONTINUE
                continue
            if back_pix == 0: #CONTINUE
                continue
        
            w2 = float(fore_pix) / im_gray.size
            w1 = float(mid_pix) / im_gray.size
            w0 = float(back_pix) / im_gray.size
        
            u2 = float(np.sum(im_gray * logic2)) / fore_pix
            u1 = float(np.sum(im_gray * logic1)) / mid_pix
            u0 = float(np.sum(im_gray * logic0)) / back_pix

            u = w1*u1 + w2*u2 + w0*u0
            g = w1*(u1-u)*(u1-u) + w0*(u0-u)*(u0-u)+w2*(u2-u)*(u2-u)

            if g_max < g:
                g_max = g
                best_th1 = threshold1
                best_th2 = threshold2
                best_u0 = u0
                best_u1 = u1
                best_u2 = u2
    return best_th1,best_th2,best_u0,best_u1,best_u2

im_gray = array(Image.open("picture/a.jpg").convert('L'))




edges = filters.gaussian(im_gray,sigma = 1)   #sigma=0.4
edges = edges * 255
th1,th2,u0,u1,u2 = OTSU(edges)




print(th1,th2,u0,u1,u2)

im_bin = np.zeros((im_gray.shape[0],im_gray.shape[1]))

for i in range(im_gray.shape[0]):
    for j in range(im_gray.shape[1]):
        if edges[i][j] >= th2:
            im_bin[i][j] = 240
        elif (edges[i][j] < th2) & (edges[i][j] >= th1):
            im_bin[i][j] = 120
        else:
            edges[i][j] = 0

figure('OTSU-3')

subplot(121)
gray()
imshow(im_bin)
axis('off')

subplot(122)
gray()
imshow(im_gray)
axis('off')

figure('figure')

hist(im_gray.flatten(),256)

show()

















