from sklearn.metrics import confusion_matrix
import numpy as np
from math import * 
def getEvalResults (matrix): 
    #ayant en argument une matrice contenant les occorances tn, fp, fn, tp
    #tn = matrix[0][0]
    #fp = matrix[0][1]
    #fn = matrix[1][0]
    #tp = matrix[1][1]
    #retour: 
    #precision , recall 
    return matrix[1][1]/(matrix[1][1]+matrix[0][1]) , matrix[1][1]/(matrix[1][1]+matrix[1][0])

def afficheScore(y_test, y_pred):
    totalMat = np.zeros((2,2));
    totalMat = totalMat + confusion_matrix(y_test, y_pred)
    precision , recall = getEvalResults(totalMat)
    f1score = 2*precision*recall / (precision+recall)
    print(str(round(precision,2))+","+str(round(recall,2))+","+str(round(f1score,2)))
    #print("  P  = ",round(precision,2))
    #print("  R  = ",round(recall,2))
    #print("  F1 = ",round(f1score,2))


#ERDE se calcule à la fin seulement, 
#en sachant à chaque pour chaque utilisateur à quelle étape on l'a classé c'est à ça que sert numPredChunk
#qui est un tableau ayant la taille de y_test (y_pred) contient pour chaque utilisateur le num de chunk dont 
#lequelle on lui a émis une décision 
def evaluateERDE(y_test, y_pred, nbWrintings, numPredChunk):  
    cfn = 1
    ctp = 1
    cfp = 0.1296  #(ce qu'ils ont pris en 2017)
    #LCo = 1- (1/(1+exp(k-o)))   k étant le nombre de soumissions du sujet concrné à cerre étape
    
    erdes5 = np.zeros(len(y_pred))
    erdes50 = np.zeros(len(y_pred))
    for i in range (len(y_pred)): 
        indices = np.arange(int(numPredChunk[i]))
        k = np.sum(nbWrintings[i][indices])
        #quand le k est très grand l'expo devient aussi très grand --> math range error 
        #pour éviter cela à partir d'un certain k, on affecte directement 1 
        #(qui est la limite de la formule des LC quand k grandit)
        if(k>300) : 
            lc5 = 1
            lc50 = 1
        else: 
            lc5 =  1- (1/(1+exp(k-5)))
            lc50 =  1- (1/(1+exp(k-50)))
            
        #la formule de calcul de ERDE: 
        if (y_pred[i]==1 and y_test[i]==-1):
            erdes5[i]=cfp
            erdes50[i]=cfp
        elif (y_pred[i]==-1 and y_test[i]==1):
            erdes5[i]=cfn
            erdes50[i]=cfn
        elif (y_pred[i]==1 and y_test[i]==1):
            erdes5[i]=lc5*ctp 
            erdes50[i]=lc50*ctp 
        else: 
            erdes5[i]=0 
            erdes50[i]=0
    #Appliquer la moyenne 
    erde5 = np.mean(erdes5)
    erde50 = np.mean(erdes50)
   
    print("ERDE5 = ", erde5)
    print("ERDE50 = ",erde50)
    
        