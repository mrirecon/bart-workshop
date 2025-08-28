import matplotlib as mpl
from matplotlib import pyplot as plt
import cfl
import numpy as np




def plotCoefficientMaps():
    fig, axs = plt.subplots(1,4, figsize=(12,4))
    axs =axs.ravel()
    img = absImageFromCFL("subspace_reco")
    mask = realImageFromCFL("roi_mask")
    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        roi = np.ma.masked_where(mask==0, img[:,:,i])
        ax.set_title("Coefficient # {}".format(i+1), fontdict={'fontsize':12})
        # ax.imshow(roi, cmap=ps.createNewMap("gray"), clim=(None, 1.2))
        ax.imshow(roi, cmap=createNewMap("gray"), clim=(None, 1.2))
    plt.show()

def plotTemporalEvolution():
    fig, axs = plt.subplots(1,4, figsize=(12,4))
    axs =axs.ravel()
    timesteps=[0,100, 200,600]
    img = absImageFromCFL("imgs")
    mask = realImageFromCFL("roi_mask")
    for i, ax in zip(timesteps,axs):
        ax.set_xticks([])
        ax.set_yticks([])
        roi = np.ma.masked_where(mask==0, img[:,:,i])
        ax.set_title("t={}".format(i), fontdict={'fontsize':12})
        ax.imshow(roi, cmap=createNewMap("gray"), clim=(None, 0.125))
    plt.tight_layout()
    plt.show()

def plotT1Map():
    fig, axs = plt.subplots(1, figsize=(4,3))
    axs.set_xticks([])
    axs.set_yticks([])
    img =absImageFromCFL("t1map")
    axs.imshow(img, cmap=createNewMap("gray"))
    plt.show()

def createNewMap(mapName):
    newmap = mpl.colormaps[mapName].copy()
    newmap.set_bad(color="black")
    if mapName == "RdBu_r":
        newmap.set_bad(color="white")
    return newmap

def realImageFromCFL(filename):
    return np.real(cfl.readcfl(filename).squeeze())
def absImageFromCFL(filename):
    return np.abs(cfl.readcfl(filename).squeeze())


def plotDict(ax, n=5):
    indx = np.random.randint(0,100000, size=n)
    filename="./subspace_dict"
    signals= realImageFromCFL(filename)[:,indx].T
    cmap=mpl.colormaps["Set1"]
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signal")
    ax.set_title("Selection from Simulated Dictionary",fontdict={'fontsize':12})
    time =np.arange(0, len(signals[0])) * 6e-3
    for i,signal in enumerate(signals):
        ax.plot(time, signal, color=cmap(i))
    
    return ax

def plotPCACoeff(ax,nCoef=4,n=30):
    ax.set_xlabel("principal component")
    ax.set_ylabel("rel. contribution")
    ax.set_title("Accumulated PCA Coefficients",fontdict={'fontsize':12})
    S = realImageFromCFL("./S")

    cumSum = np.cumsum(S)
    cumSum /= cumSum[-1]
    coeffIndx = np.arange(0,n)
    ax.plot(coeffIndx, cumSum[:n], 'ko-')
    annotation = "{} coeff. -> {:.3}% contribution".format(nCoef, cumSum[nCoef]*100)

    ax.annotate(annotation,xy=(0.1,.5), xycoords="axes fraction")

    return ax
    
    
def plotTemporalBasis(ax, n=5):
    ax.set_title("Temporal Basis",fontdict={'fontsize':12})
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signal")
    U = realImageFromCFL("./U").T[:n]
    time =np.arange(0, len(U[0])) * 6e-3
    cmap=mpl.colormaps["Set1"]
    for i,signal in enumerate(U):
        ax.plot(time, signal, color=cmap(i), label="# {}".format(i+1))

    ax.legend()


def plotSubspace():
    fig, axs = plt.subplots(1,3, figsize = (12,4))
    plotDict(axs[0],n=5)
    plotPCACoeff(axs[1], nCoef=4, n=17)
    plotTemporalBasis(axs[2],n=4)
    plt.tight_layout()
    plt.show()
    