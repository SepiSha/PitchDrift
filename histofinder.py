

#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
    
# This function finds the histogram of datapoints and also find the smoothed
# histogram using the moving average.
import numpy as np  

def histofinder(z=None):
    histogram=np.zeros(10000,'float')
    histogramSmooth=np.zeros(8000,'float')
    for i in np.arange(1-1,len(z)):
        if (z[i] > 0):
            histogram[ int(np.around(z[i])) ]=histogram[  (int(np.around(z[i])))  ] + 1   
    smoothingFactor=22
    for i in np.arange(smoothingFactor,len(histogramSmooth) - smoothingFactor):
        for j in np.arange(1,1+int(smoothingFactor / 2)): #(+1,+2...+11, ..-1,-2,,,-11)
            histogramSmooth[i]=histogramSmooth[i] + histogram[i + j] + histogram[i - j]
        histogramSmooth[i]=(histogramSmooth[i] + histogram[i]) / (smoothingFactor + 1)  
    for i in np.arange(  1-1,len(histogramSmooth)  ):
        histoStart=i
        if histogramSmooth[histoStart] > 0:
            break
    for i in np.arange(  1-1,len(histogramSmooth)-1   ):
        histoEnd=len(histogramSmooth) - i + 1-1-1
        if histogramSmooth[histoEnd] > 0:
            break
    return histogramSmooth,histoStart,histoEnd,histogram
if __name__ == '__main__':
    pass
    