# Generated with SMOP  0.41-beta
#from libsmop import *
# /Users/sepi2/Desktop/pythonStuff/smop-master/smop/m1/readPitchTimeDomain.m

#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################

# we really need to combine pitch & energy into one function.... -later -Shap
#
import numpy as np  
import scipy as scipy
from scipy import interpolate
    
#@function
def readPitchTimeDomain(data=None):
    global GusheName
    X=data[:,0].copy()
    Y_cent=data[:,1].copy()
    Y_cent3 = Y_cent / 55 + 1e-15
    Y_cent3 = Y_cent3.astype('float64')  
    Y_cent4 = np.log2(Y_cent3)
    Y_cent=np.dot(1200,Y_cent4) 
    ff=np.zeros((len(X)-1,1))
    duplicate=0
    
    for bb in range(len(X)-1):
        ff[bb]=(X[bb + 1] - X[bb])
        if (ff[bb] == 0):
            if duplicate != 0:
                raise Exception('stop:read pitch domain.---- multiple duplicates')
            duplicate=np.copy(bb)
    
    if duplicate != 0:
        X1=X[1-1:duplicate-1]
        X2=X[duplicate + 2-1:]
        X=np.concatenate((X1,X2))
        
        ff1=ff[1-1:duplicate - 1]
        ff2=ff[duplicate + 1:]
        ff=np.concatenate((ff1,ff2))
        
        Y_centtemp1=Y_cent[1-1:duplicate-1]
        Y_centtemp2=Y_cent[duplicate + 2-1:]
        Y_cent=np.concatenate((Y_centtemp1,Y_centtemp2))
    
    # fill out the missing time in the time sequence
    Ts=min(ff)
    tt=X
    f=np.copy(Y_cent)
    ttt=(tt / Ts)
    ttt = ttt.astype('float64')  
    tt  = tt.astype('float64')  
    tn=np.around(ttt)
    dt=np.diff(np.insert(tn,0,0))
    tg=np.where(dt > 1)[0]
    gaps=dt[tg] - 1
    tempvar3=np.min(tt)
    tempvar=np.amax(tt)
    tempvar2=np.amax(tn) + 1
    tempvar2=tempvar2.astype('int32')
    begining= int((tempvar3/Ts))
    ti=np.linspace(np.min(tt),tempvar,num=tempvar2-begining )
    ti2=np.linspace(0,tempvar,num=tempvar2 )[0:begining]  # from 0 to begining -1!

    #python does not like ti to start with zero when tt is >0
    #so, we had to devide it to two parts: ti & ti2 and later
    #we put them back together. -shap
    fi=scipy.interpolate.interp1d(tt,f)(ti)
    begining= int((tempvar3/Ts))
    fi2=np.empty((len(ti2)))
    fi2[:] = np.nan
    st=0   # sometimes 1 for compatibility with matlab code...
    fi = np.concatenate((fi2, fi[st:]))  #[1:] for compatibility with matlab code.
    ti = np.concatenate((ti2,ti[st:]))   # we are throwing one data out... I guess we need to change this to full array  later on
    offset=int(1)+st   # do I need st here? -shap
    for k1 in np.arange(0,len(tg)).reshape(-1):
        q=np.concatenate([np.arange(tg[k1],tg[k1] + gaps[k1] - 1)])
        fi[np.arange(offset + int(tg[k1]),offset + int(tg[k1]) + int(gaps[k1]) - 1+1)]=np.nan #(1,int(gaps[k1]))
        offset=offset + int(gaps[k1])
    X=np.copy(ti)
    fi=fi.astype('float64')
    fi[np.isnan(fi)]=0
    Y_cent=np.transpose(fi)
    # here we clear up the silence of the beginning of the audio, 
    #we do the same to X so that X and Y-cent remain the same corresponding vectors
    BeginingSilence=0
    while Y_cent[BeginingSilence] == 0:
        BeginingSilence=BeginingSilence + 1
    Y_cent=Y_cent[np.arange(BeginingSilence,len(Y_cent))]
    X=X[np.arange(BeginingSilence,len(X))]
    X=X - X[1]
    return X,Y_cent
    
if __name__ == '__main__':
    pass
    