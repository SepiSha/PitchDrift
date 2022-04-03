
#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import matplotlib.ticker as ticker
import scipy.signal as scisig
import globs
from scipy.optimize import curve_fit
           
def GaussianPeakFit(midPointRangeBegin=None,midPointRangeEnd=None,
                    i=None,z=None,histogramSmooth=None,
                    rangeBegin=None,rangeEnd=None,histoSmoothmaxlocs=None,histoStart=None,derivative=None,peakh=None,histoSmoothmaxima=None, NormalizedRangeInterval=None):


    rangeBegin          = int(rangeBegin)           
    midPointRangeEnd    = int(midPointRangeEnd)
    rangeEnd            = int(rangeEnd)
    midPointRangeBegin  = int(midPointRangeBegin )             
    
    t=(np.arange(rangeBegin,rangeEnd)).T + 1
    tMidPoint=(np.arange(midPointRangeBegin,midPointRangeEnd)).T + 1
  
    y=[]
    y=histogramSmooth[np.arange(rangeBegin,rangeEnd)]
    yMidPoint=histogramSmooth[np.arange(midPointRangeBegin,midPointRangeEnd)]
    areaunder=sum(histogramSmooth[np.arange(rangeBegin,rangeEnd)])
    
    def modelfun(x, b1, b2, b3, b4, b5):
        return b1 + np.dot(b2, x) + np.dot(b3, np.exp( -(x - b4) ** 2 / (b5)))
    b=([1,1,300,histoSmoothmaxlocs[i],10])

    ftol1 = 1e-12
    xtol1 = 1e-12
    noise_sigma = y/20000
    
    err  = -1
    err2 =-1
    err3 =-1
    err4 =-1
    exitflag = 1  # assume success for fitting
    ahatlength = 1
    try:
        popt, pcov = curve_fit(f = modelfun, xdata = t, ydata = y, p0 = b, maxfev = 180000, ftol = ftol1, xtol= xtol1, sigma=noise_sigma, absolute_sigma=True) #, p0 = [0.5,0.5,0.5,0.5,0.5])
    
        yhat = modelfun(t, *popt)
        residual= np.around(np.linalg.norm(y- yhat), 2)

        if (len(yhat) < 3):
            raise Exception('this is strange\n')
        
        bhat=scisig.find_peaks(yhat)
        if ( len(bhat[0]) == 0 ):
            ahatlength=0
            exitflag = 0
        else:
            bhat = bhat[0][0]
            ahat = y[bhat]  #y?        
            err=(sum((yhat - y) ** 2)) / len(yhat)
        
    except RuntimeError:
        exitflag = 0   

#    
# Second try: reduce range of the curve by cutting left and right
#
    
    if (err > 3 or ahatlength == 0 or (exitflag == 0)): #
        if ((histoSmoothmaxima[i] - histogramSmooth[rangeBegin] > 80) and (rangeEnd - rangeBegin > 80)):
            exitflag = 1
            rangeBegin=rangeBegin + 15
            rangeEnd=rangeEnd - 15
            t=(np.arange(rangeBegin,rangeEnd)).T + 1

            y=[]
            y=histogramSmooth[np.arange(rangeBegin,rangeEnd)]
            noise_sigma = y/20000     #it has a different range than first try. -shapour
            exitflag = 1  # assume success for fitting
            try:

                popt, pcov = curve_fit(f = modelfun, xdata = t, ydata = y, p0 = b, maxfev = 180000, ftol = ftol1, xtol= xtol1, sigma=noise_sigma, absolute_sigma=True) #, p0 = [0.5,0.5,0.5,0.5,0.5])
                fit = modelfun(t, *popt)  # optimize this the same as fist fit.
                residual= np.linalg.norm(y- modelfun(t, *popt))

                b= popt
                #why not call: yhat = modelfun(t, *popt) -shapour
                yhat=b[1-1] + np.dot(b[2-1],t) + np.dot(b[3-1],np.exp(- (t - b[4-1]) ** 2 / (b[5-1])))

                bhat=scisig.find_peaks(yhat)  # need to adampted to python
                if ( len(bhat[0]) == 0 ):
                    ahatlength=0
                    exitflag = 0
                else:
                    bhat = bhat[0][0]
                    ahat = y[bhat]  #y?
        
                    err2=(sum((yhat - y) ** 2)) / len(yhat)
                
            except RuntimeError:
                print("GausianPeakFit: Error - curve_fit second attempt failed")
                exitflag = 0   
        else:
            print("second attempt can not trim the curve.")
#
# third try: simply change initial b params and to get a better fit.
#
        if (err2 > 3 or ahatlength == 0 or (exitflag == 0)): #
            
            b=([- 2000,0.01,300,histoSmoothmaxlocs[i],8000])
            t=(np.arange(rangeBegin,rangeEnd)).T + 1
       
            y=[]
            y=histogramSmooth[np.arange(rangeBegin,rangeEnd)]
      
            exitflag = 1  # assume success for fitting
            try:
                popt, pcov = curve_fit(f = modelfun, xdata = t, ydata = y, p0 = b, maxfev = 180000, ftol = ftol1, xtol= xtol1, sigma=noise_sigma, absolute_sigma=True) #, p0 = [0.5,0.5,0.5,0.5,0.5])
    
                fit = modelfun(t, *popt)
                residual= np.linalg.norm(y- modelfun(t, *popt))
                #print( "GaussianPfit third:residule = ", residual)
            
                b= popt
                yhat=b[1-1] + np.dot(b[2-1],t) + np.dot(b[3-1],np.exp(- (t - b[4-1]) ** 2 / (b[5-1])))
                if (len(yhat) < 3):
                    raise Exception('this is strange\n')
                
                bhat=scisig.find_peaks(yhat)
                if ( len(bhat[0]) == 0 ):
                    ahatlength=0
                    exitflag = 0
                    #print("pitchdrift: bhat, third try bad")
                else:
                    bhat = bhat[0][0]
                    ahat = y[bhat]  #y?
                
                    err3=(sum((yhat - y) ** 2)) / len(yhat)
                    #print("GausianPeakFit:      SUCCESS: curve_fit 3rd try ")
                
            except RuntimeError:
                print("GausianPeakFit: Error - curve_fit third attemp failed")
                exitflag = 0   

#
# Fourth try:  no luck with gausian... let's try polynomial
#
            if (err3 > 3 or ahatlength == 0 or exitflag == 0 ):

                b=np.polyfit(t,y,2)
                yhat2=np.polyval(b,t)
                err4=(sum((yhat2 - y) ** 2)) / len(yhat2)
        
                print('pitchdrft:error2: %f \n',err2)
                yhat=yhat2
        
                bhat=scisig.find_peaks(yhat2)
                if ( len(bhat[0]) == 0 ):
                    ahatlength=0
                    raise Exception( "need to handle this case")
                bhat = bhat[0][0]
                ahat = y[bhat]  #y?
                print("GausianPeakFit:    Success: curve_fit POLY FIT")
                

    standardDev=np.std(histogramSmooth[np.arange(rangeBegin,rangeEnd+1)])
    standardDev2=np.std(histogramSmooth[np.arange(midPointRangeBegin,midPointRangeEnd+1)])

    xlength=rangeEnd - rangeBegin
    ylength=max(histogramSmooth[np.arange(rangeBegin,rangeEnd)]) - min(histogramSmooth[np.arange(rangeBegin,rangeEnd)])
    Slope=ylength / (np.dot(2,xlength))
    xrange=np.concatenate([[rangeBegin - 60,rangeEnd + 60]])
    yrange=([min(histogramSmooth[np.arange(rangeBegin,rangeEnd)]) - 2,max(histogramSmooth[np.arange(rangeBegin,rangeEnd)]) + 2])
    ymax=max(np.concatenate( ( y, yMidPoint )    ))

    if max(yhat) > yrange[2-1]:
        yrange[2-1]=max(yhat)

    if (ahatlength == 0):   # No peak found with Gaussian and polynomial fit
        print("GaussianDebug")
        bhat=(rangeEnd - rangeBegin) / 2
        ahat=yhat[round(bhat)]

    fittedPeak=bhat + rangeBegin

    if Slope > 0.7:
        xrange=xrange + [- 50,+ 50]

    return yhat, fittedPeak