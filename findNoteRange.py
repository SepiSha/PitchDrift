
# I have modified this function and checked it with MATLLAB (which is the best version) 
#in Jan 30, 2021
# Generated with SMOP  0.41-beta
#from libsmop import *
# /Users/sepi2/Desktop/pythonStuff/smop-master/smop/m1/findNoteRange.m

#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
    
# this function finds the range of each mountain in histogram, and then pass the range of the mountain to
# a function to fit a Gaussian to that mountain and finds the peak of the
# gaussian as the audio pitch estimate of each note.
import numpy as np  
import GaussianPeakFit as GaussianPeakFit  
import matplotlib.pyplot as plt    


def findNoteRange(histogramSmooth=None,z=None,derivative=None,histoSmoothmaxima=None,
                  histoSmoothmaxlocs=None,histoStart=None):
    
    if len(histoSmoothmaxlocs)==0:
        raise Exception('findNoteRange--histoSmoothmaxlocs is 0')
    
    DEBUGFLAG=False
    rangeBegin = np.zeros(0)  # start with empty lists, append to them in i loop
    rangeEnd = np.zeros(0)
    RangeInterval = np.zeros(0)
    NormalizedRangeInterval=np.zeros(0)
    for i in np.arange(1-1,len(histoSmoothmaxlocs)): #Peak loop
    # first we move from the points on top of the peak to the left by 20, to
    # get rid of the noises on top
        if histoSmoothmaxlocs[i] < histoStart:
            raise Exception('findNoteRange: error1 histoSmoothmaxlocs[i)=%d  <  %d ',histoSmoothmaxlocs[i],histoStart)
            continue
        j=histoSmoothmaxlocs[i] - histoStart - 20
        
        if (j < 1):
            j=1

        # Takes the 20# around the peak from the left side
        peakh=(0.8*histoSmoothmaxima[i])
        while (histogramSmooth[j + histoStart + 7] > (0.8*histoSmoothmaxima[i])):
            if (histogramSmooth[j + histoStart] < histogramSmooth[j + histoStart - 40]):
                break
            j=j - 1
            if j < 2:
                break
        if j < 1:
            j=1
        while ((derivative[j-1]) > 1.9):  # Matlab->python -1 array compensation

            if j > 1:
                j=j - 1
            else:
                break

        rangeBegin= np.append(rangeBegin, [j + histoStart + 7] )
 
        ##################################################
        #Starting to find rangeEnd
        ##################################################
        
        k=histoSmoothmaxlocs[i] - histoStart + 20

        if (k > len(derivative)-1):
            print("findNoteRange: why would this happen?", k, len(derivative))
            k=len(derivative)-1
            
        while (derivative[k-1] < -1.9):

            if (k == len(derivative)-1):
                break
            k=k + 1

        if (k > len(derivative)):
            k=len(derivative)
        while (histogramSmooth[k + histoStart + 7] > (0.8*histoSmoothmaxima[i])):

            if k == len(derivative):
                break
            k=k + 1

        while (derivative[k-1] < - 1.5):

            if (k == len(derivative)):
                break
            k=k + 1

        if (histogramSmooth[k + histoStart + 7] < histogramSmooth[k + histoStart]):
            rangeEnd= np.append(rangeEnd, [k + histoStart + 7] )

        else:
            rangeEnd= np.append(rangeEnd, [k + histoStart] )
   
        ###################################################################
        # Extend the range if the height of the beginning and end of histogram are too different
        
        #if (- histogramSmooth[int(rangeEnd[i])] + histogramSmooth[int(rangeBegin[i])] > 0.4):
        while (histogramSmooth[int(rangeBegin[i])] - histogramSmooth[int(rangeBegin[i]) - 15] > 0.2) & (- histogramSmooth[int(rangeEnd[i])] + histogramSmooth[int(rangeBegin[i])] > 0.4):

            rangeBegin[i]=rangeBegin[i] - 7

        #if (histogramSmooth[int(rangeEnd[i])] - histogramSmooth[int(rangeBegin[i])] > 0.4):
        while (histogramSmooth[int(rangeEnd[i])] - histogramSmooth[int(rangeEnd[i]) + 15] > 0.2) & (histogramSmooth[int(rangeEnd[i])] - histogramSmooth[int(rangeBegin[i])] > 0.4):

            if rangeEnd[i] > len(histogramSmooth):
                break
            rangeEnd[i]=rangeEnd[i] + 7


        #make the left and right side 'almost' equal
        while ((histogramSmooth[int(rangeEnd[i])] - histogramSmooth[int(rangeBegin[i])]) > 0.4):

            rangeBegin[i]=rangeBegin[i] + 1

      
        #shorten right side
        while (- histogramSmooth[int(rangeEnd[i])] + histogramSmooth[int(rangeBegin[i])] > 0.4):
            #print("findNote3:",- histogramSmooth[int(rangeEnd[i])], histogramSmooth[int(rangeBegin[i])],int(rangeBegin[i]), int( rangeEnd[i]))
            rangeEnd[i]=rangeEnd[i] - 1
      
        
        # one more time, let's try to find a nice curve!
        # cut the flat ends from the sides:
        
        if (histoSmoothmaxima[i] <20 ):
            leftSlope=0.90
            rightSlope=0.40
        if (histoSmoothmaxima[i] <7):
            leftSlope=0.50
            rightSlope=0.30
        if (histoSmoothmaxima[i] >=20 ):           # >20
            leftSlope=1.50
            rightSlope=0.90  
        for p in np.arange(1-1,5):
            if (rangeEnd[i] - rangeBegin[i] < 90): #range is too short
                break
            
            if (histogramSmooth[int(rangeBegin[i]) + 7] - histogramSmooth[int(rangeBegin[i])]) < leftSlope:
                rangeBegin[i]=rangeBegin[i] + 7

            if (histogramSmooth[int(rangeEnd[i]) - 7] - histogramSmooth[int(rangeEnd[i])]) < rightSlope:
                rangeEnd[i]=rangeEnd[i] - 7
                
            if (histogramSmooth[int(rangeEnd[i]) - 7] - histogramSmooth[int(rangeEnd[i])]) <0 : # going up mistakenly!
                rangeEnd[i]=rangeEnd[i] - 7
                #raise Exception('findNoteRange: error2 should not happen ')
            

        #if the curve is tall enough, sometime the Begin and end cause fitting
        #problem, here we reduce that chance:
        #change this if to while, walk slowly
        if ((histoSmoothmaxima[i] - histogramSmooth[int(rangeBegin[i])] > 80) and (rangeEnd[i] - rangeBegin[i] > 80)):
            rangeBegin[i]=rangeBegin[i] + 15

            rangeEnd[i]=rangeEnd[i] - 15

        if (rangeEnd[i] - rangeBegin[i]) < 3:
            continue
        if (i < len(histoSmoothmaxlocs)-1):
            if (   rangeEnd[i] > histoSmoothmaxlocs[i + 1]   ): #strange case that rangeEnd of one peak has passed the next peak
                x=np.argmin(   histogramSmooth[np.arange(histoSmoothmaxlocs[i],histoSmoothmaxlocs[i + 1])]    )
                rangeEnd[i]=x + histoSmoothmaxlocs[i]
        
        maxx=np.max(histogramSmooth[int(rangeBegin[i]):int(rangeEnd[i])])
        minn=np.min(histogramSmooth[int(rangeBegin[i]):int(rangeEnd[i])])
        RangeInterval=np.append(RangeInterval,maxx-minn)
        NormalizedRangeInterval=np.append(NormalizedRangeInterval,(maxx-minn)/(maxx+minn))
        
 
        # A new way to find range: take the range to be from half-height of the montain
        # find the right and left point of the curve with height=1/2peak, we draw these lines and 
        # don't use it for other reasons
        locpeakNEW=histoSmoothmaxlocs[i]

        midPointRangeEnd=rangeEnd[i]
        
        tempFlag0=False
        for mm1 in np.arange(locpeakNEW,locpeakNEW + 100): 

            if (histogramSmooth[mm1] < histogramSmooth[int(locpeakNEW)] / 2): 
                if (histogramSmooth[mm1]<histogramSmooth[int(midPointRangeEnd)]):
                    midPointRangeEnd=mm1
                    tempFlag0=True
            if (histogramSmooth[mm1] < histogramSmooth[int(locpeakNEW)] / 3): 
                if (histogramSmooth[mm1]<histogramSmooth[int(midPointRangeEnd)]):
                    midPointRangeEnd=mm1
                    tempFlag0=True
            if (histogramSmooth[mm1] < histogramSmooth[int(locpeakNEW)] / 4): 
                if (histogramSmooth[mm1]<histogramSmooth[int(midPointRangeEnd)]):
                    midPointRangeEnd=mm1
                    tempFlag0=True
                break
            

        if tempFlag0==False:
            if (histogramSmooth[mm1]<histogramSmooth[int(midPointRangeEnd)]):
                midPointRangeEnd=mm1

        tempFlag=False
        midPointRangeBegin=rangeBegin[i]
        for mm2 in np.arange(locpeakNEW,locpeakNEW - 100,- 1).reshape(-1): 

            if (histogramSmooth[mm2] < histogramSmooth[locpeakNEW] / 2): 
                if (histogramSmooth[mm2]<histogramSmooth[int(midPointRangeBegin)]):
                    midPointRangeBegin=mm2
                    tempFlag=True
            if (histogramSmooth[mm2] < histogramSmooth[locpeakNEW] / 3): 
                if (histogramSmooth[mm2]<histogramSmooth[int(midPointRangeBegin)]):
                    midPointRangeBegin=mm2
                    tempFlag=True
            if (histogramSmooth[mm2] < histogramSmooth[locpeakNEW] / 4):
                if (histogramSmooth[mm2]<histogramSmooth[int(midPointRangeBegin)]):
                    midPointRangeBegin=mm2
                    tempFlag=True
                break

        if tempFlag==False:
            if (histogramSmooth[mm2]<histogramSmooth[int(midPointRangeBegin)]):
                midPointRangeBegin = mm2
            

        #midPointRangeBegin     histoSmoothmaxlocs[i]   midPointRangeEnd   -- 
        
        minHist2=5000
        delta=5
        
        tmprangeBegin = midPointRangeBegin
        #for tr in np.arange(midPointRangeBegin+delta  ,  midPointRangeBegin+int((int(histoSmoothmaxlocs[i])-midPointRangeBegin)/3) ):
        for tr in np.arange(midPointRangeBegin+delta  ,  midPointRangeBegin+int((int(histoSmoothmaxlocs[i])-midPointRangeBegin)) ):

            if ( histogramSmooth [int(tr)] <= minHist2):
                tmprangeBegin = tr
                minHist2=histogramSmooth [int(tr)]
        m1=  abs((histogramSmooth[int(locpeakNEW)]     -histogramSmooth[int(midPointRangeBegin)]) /(locpeakNEW-midPointRangeBegin))
        m2=  abs((histogramSmooth[int(locpeakNEW)]-histogramSmooth[int(tmprangeBegin)]) /(locpeakNEW-tmprangeBegin))

        if (( midPointRangeBegin +delta+5 < tmprangeBegin ) and  (m2>m1) ): 
            midPointRangeBegin = tmprangeBegin

        minHist2=5000

        # Above we took care of a small extra hill above. now we try to
        # remove any upgoing tail at the begining.  
        
        tmprangeBegin = midPointRangeBegin
        #For some reason we used to go upto 1/3 of the height but doesn't work
        #if tail goes very high Nov 2021
        for tr in np.arange(midPointRangeBegin  ,  midPointRangeBegin+int((int(histoSmoothmaxlocs[i])-midPointRangeBegin)) ): #int(histoSmoothmaxlocs[i])-5
            if ( histogramSmooth [int(tr)] <= minHist2):
                tmprangeBegin = tr
                minHist2=histogramSmooth [int(tr)]
        
        if ( midPointRangeBegin  < tmprangeBegin ) : 
            midPointRangeBegin = tmprangeBegin
        
        minHist2=5000
        delta=5
        tmprangeEnd =midPointRangeEnd
        for tr in np.arange(  midPointRangeEnd-delta, int(histoSmoothmaxlocs[i])+ int((2*(midPointRangeEnd-int(histoSmoothmaxlocs[i])))/3),-1):
            if ( histogramSmooth [int(tr)] <= minHist2): 
                tmprangeEnd = tr
                minHist2=histogramSmooth [int(tr)]
        m1=  abs((histogramSmooth[int(locpeakNEW)]     -histogramSmooth[int(midPointRangeEnd)]) /(locpeakNEW-midPointRangeEnd))
        m2=  abs((histogramSmooth[int(locpeakNEW)]-histogramSmooth[int(tmprangeEnd)]) /(locpeakNEW-tmprangeBegin))

        if ( (midPointRangeEnd +delta+5 < tmprangeEnd ) and (m2>m1))   : 
            midPointRangeEnd = tmprangeEnd
            
        # Above we took care of a small extra hill above. now we try to
        # remove any upgoing tail at the begining.  --shapour
        tmprangeEnd = midPointRangeEnd
        minHist2=5000
        for tr in np.arange(midPointRangeEnd, int(histoSmoothmaxlocs[i])+int((2*(midPointRangeEnd-int(histoSmoothmaxlocs[i])))/3), -1 ):
            if ( histogramSmooth [int(tr)] <= minHist2):
                tmprangeEnd = tr
                minHist2=histogramSmooth [int(tr)]
        
        if ( midPointRangeEnd  < tmprangeEnd ) : 
            midPointRangeEnd = tmprangeEnd
        
          
        minHist2=5000
        delta=10
        
        tmprangeBegin = rangeBegin[i]
        for tr in np.arange(int(rangeBegin[i])+delta  ,  int(rangeBegin[i])+int((int(histoSmoothmaxlocs[i])-int(rangeBegin[i]))/3) ):
            if ( histogramSmooth [tr] <= minHist2):
                tmprangeBegin = tr
                minHist2=histogramSmooth [tr]
        m1=  abs((histogramSmooth[locpeakNEW]     -histogramSmooth[int(rangeBegin[i])]) /(locpeakNEW-rangeBegin[i]))
        m2=  abs((histogramSmooth[int(locpeakNEW)]-histogramSmooth[int(tmprangeBegin)]) /(locpeakNEW-tmprangeBegin))

        if (( rangeBegin[i] +delta+5 < tmprangeBegin ) and  (m2>m1) ): 
            rangeBegin[i] = tmprangeBegin
                
        # Above we took care of a small extra hill above. now we try to
        # remove any upgoing tail at the begining.  --shapour
        tmprangeBegin = rangeBegin[i]
        for tr in np.arange(int(rangeBegin[i])  ,  int(rangeBegin[i])+int((int(histoSmoothmaxlocs[i])-int(rangeBegin[i]))/3) ):
            if ( histogramSmooth [tr] <= minHist2):
                tmprangeBegin = tr
                minHist2=histogramSmooth [tr]
        
        if ( rangeBegin[i]  < tmprangeBegin ) : 
            rangeBegin[i] = tmprangeBegin   
        

        minHist2=5000
        delta=5
        tmprangeEnd =rangeEnd[i]
        for tr in np.arange(  int(rangeEnd[i])-delta, int(histoSmoothmaxlocs[i])+int(((int(rangeEnd[i])-int(histoSmoothmaxlocs[i])))) ,-1):
            if ( histogramSmooth [tr] <= minHist2): #histogramSmooth[int(rangeEnd[i])]):
                tmprangeEnd = tr
                minHist2=histogramSmooth [tr]
        m1=  abs((histogramSmooth[int(locpeakNEW)]     -histogramSmooth[int(midPointRangeEnd)]) /(locpeakNEW-midPointRangeEnd))
        m2=  abs((histogramSmooth[int(locpeakNEW)]-histogramSmooth[int(tmprangeEnd)]) /(locpeakNEW-tmprangeBegin))

        if ( (rangeEnd[i] +delta+5 < tmprangeEnd ) and (m2>m1))   : 
            rangeEnd[i] = tmprangeEnd
  

        # Above we took care of a small extra hill above. now we try to
        # remove any upgoing tail at the begining.  
        tmprangeEnd = rangeEnd[i]
        minHist2 = 5000   # added Aug 19
        for tr in np.arange(int(rangeEnd[i]), int(histoSmoothmaxlocs[i])+int(((int(rangeEnd[i])-int(histoSmoothmaxlocs[i])))),  -1 ):
            if ( histogramSmooth [tr] <= minHist2):
                tmprangeEnd = tr
                minHist2=histogramSmooth [tr]
        
        if ( rangeEnd[i]  > tmprangeEnd ) : 
            rangeEnd[i] = tmprangeEnd
            
            
        if (NormalizedRangeInterval[i] < 0.44): #Sep changed 0.4 to 0.45
            rangeBegin[i]=midPointRangeBegin;
            rangeEnd[i]=midPointRangeEnd;
            
       
        yhat,histoSmoothmaxlocs[i]=GaussianPeakFit.GaussianPeakFit(midPointRangeBegin,midPointRangeEnd,i,z,histogramSmooth,rangeBegin[i],rangeEnd[i],histoSmoothmaxlocs,histoStart,derivative,peakh,histoSmoothmaxima,NormalizedRangeInterval[i])

           
    # find the max area under mountans for shahed, find anothe method for spare shahed, in case the first
    # one dosn't work well
    histoArea = np.zeros(len(histoSmoothmaxlocs))
    for i in np.arange(1-1,len(histoSmoothmaxlocs)):
        histoArea[i]=np.sum(histogramSmooth[int(rangeBegin[i]):int(rangeEnd[i])+1])

    maxArea = histoArea.max(0)
    I = histoArea.argmax(0)

    audioShahed=histoSmoothmaxlocs[I]

    for i in np.arange(1-1,len(histoSmoothmaxlocs)):
        histoArea[i]=sum(histogramSmooth[np.arange(histoSmoothmaxlocs[i] - 55,histoSmoothmaxlocs[i] + 55)])

    maxArea = histoArea.max(0)
    I = histoArea.argmax(0)

    audioShahedSpare=histoSmoothmaxlocs[I]
    return audioShahed,audioShahedSpare,histoSmoothmaxlocs