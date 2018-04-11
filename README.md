# OTSU

This project based on the “Digital Image Processing” course.

OTSU is an image binarization algorithm using Fisher's Discriminant Analysis.

Resource: https://en.wikipedia.org/wiki/Otsu%27s_method

Others: During the coding, I found that there is few example about the OTSU multi-classification. So I just tried my best to reason the formulas to achieve the three classification.

Two files: 

#==================================================

(using b.jpg)

1.	Two_classification:

    Divide the picture into two different parts: foreground pixels and background pixels.

    # Result: 

    The first picture is the binary image without using Gaussian Filter.

    Threshold = 117


    Second one is the binary image after using Gaussian Filter. (sigma = 1)

    Threshold = 121


    From the histogram, we can find b.jpg actually has four levels of depth. But we just classify it into two class. (The top of the curve is the best threshold)

2.	Three_classification:

    Result:

    Threshold = 100, 185

    The top of the surface is the best threshold
