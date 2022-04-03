# Generated with SMOP  0.41-beta
#from libsmop import *
# /Users/sepi2/Desktop/pythonStuff/smop-master/smop/m1/extrapolateZeros.m

    #########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
    
# In the process of pitch recognition, there are smalle intervals of unrecognized frequencies
# We have extrapolated those frequencies with linear regression, using the
# frequencies around them.

import numpy as np 
import globs

def extrapolateZeros(Y_cent=None):

    Y_centExtrapolated=np.copy(Y_cent)
    i=10
    gaplen=35
    
    while (i < len(Y_cent) - 8):
        i=i + 1
        if (Y_cent[i] == 0):
            b=(i)
            while (Y_cent[i] == 0):
                i=i + 1
                if (i > (len(Y_cent))):
                    break
            if (i > (len(Y_cent))):
                e=i - 1
            else:
                e=(i)
            if ((abs(Y_cent[e] - Y_cent[b - 1]) < 500) and  (e - b < gaplen) ): 
                for j in np.arange(b,e).reshape(-1):
                    Y_centExtrapolated[j]=Y_cent[b - 1] + np.dot((Y_cent[e] - Y_cent[b - 1]),(j - b)) / (e - b)
    return Y_centExtrapolated