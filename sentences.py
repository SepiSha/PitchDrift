#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
    
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import globs
from dataclasses import dataclass    

@dataclass
class SEG:
    curveBeg: int
    curveEnd: int
    zeroEnd: int

def sentences(y_cent, name):

    i=1-1
    while (y_cent[i] == 0):
        i=i + 1
    j=1-1
    seg=[]
    while (i <  len(y_cent)-1):
        seg = np.append(seg, SEG(-1,-1,-1))
        seg[j].curveBeg = (i)
        seg[j].curveEnd = (- 1)
        seg[j].zeroEnd  = (- 1)
        while (y_cent[i] != 0) and not(np.isnan(y_cent[i])):
            if (i <  len(y_cent)-1):
                i=i + 1
            else:
                break

        if (i >=  len(y_cent)):
             seg[j].curveEnd = ( len(ix))
             break
        seg[j].curveEnd = (i)

        while ((y_cent[i] == 0) or np.isnan(y_cent[i])):
            i=i + 1
            if (i >  len(y_cent)):
                break

        seg[j].zeroEnd = (i)

        if  seg[j].zeroEnd >  len(y_cent):
             seg[j].zeroEnd = ( seg[j].zeroEnd - 1)
        j=j + 1
    

    i=1-1

    while i <  len(seg)-1:

        if ( seg[i].zeroEnd -  seg[i].curveEnd < 40): 
            seg[i].curveEnd = ( seg[i + 1].curveEnd)
            seg[i].zeroEnd = ( seg[i + 1].zeroEnd)
            seg=np.concatenate([ seg[np.arange(1-1,i)], seg[np.arange(i + 2-1,len(seg))]    ])

        else:
            i=i + 1

  
    i=1-1
    doIt=0
    notYet=1000
    while i <  len(seg)-1:
        doIt=0
        inc=1

        if ((( seg[i].curveEnd -  seg[i].curveBeg < 600) and ( seg[i].zeroEnd -  seg[i].curveEnd < 150)) or ( seg[i].curveEnd -  seg[i].curveBeg < 50)):
            seg[i].curveEnd = ( seg[i + 1].curveEnd)
            seg[i].zeroEnd = ( seg[i + 1].zeroEnd)
            inc=0
            seg= np.concatenate ((seg[1-1:i], seg[i + 2:len(seg)-1]))

        else:
            if (5==6):
                notYet=0
        if ((doIt == 4)):
            seg[i].curveEnd = ( seg[i + 1].curveEnd)
            seg[i].zeroEnd = ( seg[i + 1].zeroEnd)
            inc=0

            seg=np.concat([ seg[np.arange(1,i)], seg[np.arange(i + 2,len(seg))]])
        i=i + inc

     
    for i in np.arange(2-1, len(seg)):  # from second segment, go back if there is space
        ii=1
        while (y_cent[ seg[i].curveBeg - 1] == 0) and (ii < 50) and (seg[i].curveBeg <  len(y_cent)):
            seg[i].curveBeg = ( seg[i].curveBeg - 1)
            ii=ii + 1

        ii=1
        while ((seg[i].curveEnd <  len(y_cent)-1) and y_cent[seg[i].curveEnd + 1] == 0) and (ii < 30):
            seg[i].curveEnd = ( seg[i].curveEnd + 1)
            ii=ii + 1

        if (ii < 3) and (i !=  len(seg)-1):
            plt.figure()
            plt.plot(y_cent)
            plt.title('problematic curve????')
            raise Exception('++splitplot: ..ii < 3')

    name ="splitplot+"+name
    return seg
if __name__ == '__main__':
    pass

