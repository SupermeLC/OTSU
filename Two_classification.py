from PIL import Image
from pylab import *
from skimage import filters
import numpy as np

def OTSU(im_gray):
    g_max = 0
    best_th = 0
    G = np.zeros(256)

    for threshold in range(0,256,1):
        logic0 = im_gray > threshold
        logic1 = im_gray <= threshold
        
        fore_pix = np.sum(logic0)
        back_pix = np.sum(logic1)
        if fore_pix == 0: #END
            break
        if back_pix == 0: #CONTINUE
            continue
        
        w0 = float(fore_pix) / im_gray.size
        w1 = float(back_pix) / im_gray.size
        
        u0 = float(np.sum(im_gray * logic0)) / fore_pix
        u1 = float(np.sum(im_gray * logic1)) / back_pix


        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        
        G[threshold] = g
        if g_max < g:
            g_max = g
            best_th = threshold
            best_u0 = u0
            best_u1 = u1
    return best_th,best_u0,best_u1,G


im_gray = array(Image.open("picture/b.jpg").convert('L'))

th,u0,u1,k = OTSU(im_gray)#原图
print(th)
print(k[th])
im_bin1 = 255 * (im_gray >= th)


edges = filters.gaussian(im_gray,sigma = 1)   #sigma参数
edges = edges * 255
th,u0,u1,G = OTSU(edges)#高斯滤波后图片
print(th)
print(G[th])
im_bin2 = 255 * (edges >= th)
x = np.arange(0,256,1);

figure('OTSU-2')

subplot(131)
gray()
imshow(im_bin1)
axis('off')

subplot(132)
gray()
imshow(im_bin2)
axis('off')

subplot(133)
gray()
imshow(edges)
axis('off')

figure("Hist")
hist(edges.flatten(),256)
plot(x, G,'r')

show()





