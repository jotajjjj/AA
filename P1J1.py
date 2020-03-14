import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
#para graficar en 3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




def grafica3D():
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    # make data
   # X=np.arange(-5,5,0.25)
    #Y=np.arange(-5,5,0.25)
    #X,Y=np.meshgrid(X,Y)
    #R=np.sqrt(X**2+Y**2)
    #Z=np.sin(R)


    Datos=make_data(np.array([-10,10]),np.array([-1,4]),X,Y)

    surf=ax.plot_surface(Datos[0],Datos[1],Datos[2],cmap=cm.coolwarm,
                linewidth=0,antialiased=False)


    ax.set_zlim(0,700)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf,shrink=0.5,aspect=5)

    plt.show()

def make_data(t0_range,t1_range,X,Y):
    step=0.1

    ## Theta0 será la base 
    Theta0=np.arange(t0_range[0],t0_range[1],step)
    
    Theta1=np.arange(t1_range[0],t1_range[1],step)

    Theta0,Theta1=np.meshgrid(Theta0,Theta1)

    Coste=np.empty_like(Theta0)

    for ix,iy in np.ndindex(Theta0.shape):
        Coste[ix,iy]=coste(X,Y,[Theta0[ix,iy],Theta1[ix,iy] ])

    return [Theta0,Theta1,Coste]



def carga_csv(file_name):
    valores=read_csv(file_name,header=None).values
    return valores.astype(float) 

def coste(X,Y,Theta):
    
    H=np.dot(X,Theta)
    Aux=(H-Y)**2
    return Aux.sum()/(2*len(X))

def gradiente(X,Y,Theta,alpha):
    
    NuevaTheta=Theta
    m=np.shape(X)[0]#numero de filas
    n=np.shape(X)[1]#numero de columnas
    H=np.dot(X,Theta)#hipotesis THETA traspuesta *X  
    Aux=(H-Y)#valor predicho menos valor real

    #FunCoste=coste(X,Y,NuevaTheta)
    #print(FunCoste)


    for i in range(n):
        Aux_i=Aux*X[:,i]
        NuevaTheta[i]-=(alpha/m)*Aux_i.sum()
    return NuevaTheta




datos=carga_csv('ex1data1.csv')

Ini=datos[:,:-1] #columna variable independiente inicial
Fin=datos[:,-1]  #columna variable dependiente inicial

X=datos[:,:-1]#quitamos la ultima columna

Y=datos[:,-1]#seleccionamos la ultima columna  

#print(Y)
np.shape(Y)

m=np.shape(X)[0]
n=np.shape(X)[1]

X=np.hstack([np.ones([m,1]),X])

Theta=np.array([0.01,0.001]) #vector de tethas
Gra=Theta
for i in range(1500):
    alpha=0.01
    Theta=gradiente(X,Y,Theta,alpha)
    X2=datos[:,0]
    Ini=datos[:,:-1]
    
  
#Gra=np.dot(X,Theta)
#plt.plot(Ini,Gra,'r')
#plt.plot(Ini,Fin,'x')
#plt.show()

#necesitamos la 1 columa y la segunda columna de Thetas por separado

#Th1=Theta[0]
#Th2=Theta[1]

#Datos=make_data(np.array([-10,10]),np.array([-1,4]),X,Y)


grafica3D()
