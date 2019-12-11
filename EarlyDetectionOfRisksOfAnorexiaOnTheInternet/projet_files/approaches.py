from classifier import *
from parser import * 
from evaluator import * 
def approach1(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
	print("APPROCHE 1: Prédire pour un chunk k du corpus de test en apprenant seulement sur le chunk k du corpus d'apprentissage ")
	print("On ne fait pas la distiction entre les soumissions dans un chunk")
	print("ie: On aura que la taille de X est égale aux nombre de subjects")
	for i in range (1,nbChunks+1) : 
	    print("\n\nCHUNK ",i)
	    #On essaie une approche ou si on veut predire pour un chunk k en test, on apprend seulement sur le chunk k. 
	    Xtrain, Ytrain, Xtest, Ytest = extractingChunkText (trainSubjects, trainLabels, testSubjects,testLabels,i) 
	    applyingModels(Xtrain, Ytrain, Xtest, Ytest)


def approach2(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
	print("APPROCHE 2: le nombre de chunks en apprentissage est égal à celui en test")
	print("C'est à dire si on prédit en ayant 2 chunks en test par exemple")
	print("on entrainera notre modèle seulement sur les deux premiers chunks ")
	print("On ne fait pas la distiction entre les soumissions dans un chunk")
	print("ie: On aura que la taille de X est égale aux nombre de sujets")
	for i in range (1,nbChunks+1) : 
	    print("\n\nCHUNK ",i)
	    Xtrain, Ytrain, Xtest, Ytest = extractingNChunkText (trainSubjects, trainLabels, testSubjects,testLabels,i) 
	    applyingModels(Xtrain, Ytrain, Xtest, Ytest)

def approach3(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
	print("APPROCHE 3: Prédire pour un chunk k du corpus de test en apprenant seulement sur le chunk k du corpus d'apprentissage")
	print("On fait la distiction entre les soumissions dans un chunk")
	print("ie: On aura que la taille de X est égale aux nombre de sujets * le nombre moyens de soumissions par sujet")
	for i in range (1,nbChunks+1) : 
	    print("\n\nCHUNK ",i)
	    Xtrain, Ytrain, Xtest, Ytest = extractingWritingsFromChunk (trainSubjects, trainLabels, testSubjects,testLabels,i) 
	    applyingModels(Xtrain, Ytrain, Xtest, Ytest)


def approach4(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
	print("APPROCHE 4: le nombre de chunks en apprentissage est égal à celui en test")
	print("C'est à dire si on prédit en ayant 2 chunks en test par exemple")
	print("on entrainera notre modèle seulement sur les deux premiers chunks ")
	print("On fait la distiction entre les soumissions dans un chunk")
	print("ie: On aura que la taille de X est égale aux nombre de sujets * le nombre moyens de soumissions par sujet")
	for i in range (1,nbChunks+1) : 
	    print("\n\nCHUNK ",i)
	    Xtrain, Ytrain, Xtest, Ytest = extractingWritingsFromNChunk (trainSubjects, trainLabels, testSubjects,testLabels,i) 
	    applyingModels(Xtrain, Ytrain, Xtest, Ytest)


def approach2TresholdRegLogLsaTfIdf (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
    
   
    lastPred= np.zeros(len(testLabels))
    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    numPredChunk = np.zeros(len(testLabels))
    #Il faudrait penser à un moyen pour stocker le nombre de soumissions(writings) de chaque utilisateur 
    #après chaque chunk. 
    #fait dans la fct extractingNChunkText2 qui renvoie 
    #aussi le nombre de writings de chaque sujet de test dans chaque chunk (un tableau 2D)
    for i in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",i)
        Xtrain, Ytrain, Xtest, Ytest,nbWritings = extractingNChunkText2 (trainSubjects, trainLabels, testSubjects,testLabels,i)    
        lastPred, numPredChunk, y_pred  = applyingRegLogLsaTfIdfThreshold(Xtrain, Ytrain, Xtest, Ytest, i,numPredChunk,lastPred)
    evaluateERDE(Ytest, y_pred, nbWritings, numPredChunk)
   


def approach2TresholdCnnLsaTfIdf (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :
    
    lastPred= np.zeros(len(testLabels))
    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    numPredChunk = np.zeros(len(testLabels))
    #Il faudrait penser à un moyen pour stocker le nombre de soumissions(writings) de chaque utilisateur 
    #après chaque chunk. 
    #fait dans la fct extractingNChunkText2 qui renvoie 
    #aussi le nombre de writings de chaque sujet de test dans chaque chunk (un tableau 2D)
    for i in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",i)
        Xtrain, Ytrain, Xtest, Ytest,nbWritings = extractingNChunkText2 (trainSubjects, trainLabels, testSubjects,testLabels,i)    
        lastPred, numPredChunk, y_pred  = applyingCnnLsaTfIdfThreshold(Xtrain, Ytrain, Xtest, Ytest, i,numPredChunk,lastPred)
    evaluateERDE(Ytest, y_pred, nbWritings, numPredChunk)
    




def approach2TresholdCnnDocToVec (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :


    lastPred= np.zeros(len(testLabels))
    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    numPredChunk = np.zeros(len(testLabels))
    #Il faudrait penser à un moyen pour stocker le nombre de soumissions(writings) de chaque utilisateur 
    #après chaque chunk. 
    #fait dans la fct extractingNChunkText2 qui renvoie 
    #aussi le nombre de writings de chaque sujet de test dans chaque chunk (un tableau 2D)
    for i in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",i)
        Xtrain, Ytrain, Xtest, Ytest,nbWritings = extractingNChunkText2 (trainSubjects, trainLabels, testSubjects,testLabels,i)    
        lastPred, numPredChunk, y_pred= applyingCnnDocToVecThreshold(Xtrain, Ytrain, Xtest, Ytest, i,numPredChunk,lastPred)
    
    evaluateERDE(Ytest, y_pred, nbWritings, numPredChunk)




def approach2TresholdRegLogDocToVec (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) :

    lastPred= np.zeros(len(testLabels))
    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    numPredChunk = np.zeros(len(testLabels))
    #Il faudrait penser à un moyen pour stocker le nombre de soumissions(writings) de chaque utilisateur 
    #après chaque chunk. 
    #fait dans la fct extractingNChunkText2 qui renvoie 
    #aussi le nombre de writings de chaque sujet de test dans chaque chunk (un tableau 2D)
    for i in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",i)
        Xtrain, Ytrain, Xtest, Ytest,nbWritings = extractingNChunkText2 (trainSubjects, trainLabels, testSubjects,testLabels,i)    
        lastPred, numPredChunk, y_pred= applyingRegLogDocToVecThreshold(Xtrain, Ytrain, Xtest, Ytest, i,numPredChunk,lastPred)
    
    evaluateERDE(Ytest, y_pred, nbWritings, numPredChunk)


def approachtimeSeriesLsaTfIdfLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks): 

    for numChunk in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",numChunk)
        trainTimeSeries = extractTimeSeries(trainSubjects,numChunk) 
        testTimeSeries = extractTimeSeries(testSubjects,numChunk) 
        Ytrain = extractingLabels(trainSubjects,trainLabels)
        Ytest = extractingLabels(testSubjects,testLabels)
        Ytrain = np.where(Ytrain==1, 1, 0)
        Ytest = np.where(Ytest==1, 1, 0)
        
        trainLSATimeSeries , testLSATimeSeries = processTfIdfLSATimeSeries(trainTimeSeries,testTimeSeries)
        # reshape input to be 3D [samples, timesteps, features]
        train_X = trainLSATimeSeries.reshape((trainLSATimeSeries.shape[0], numChunk, trainLSATimeSeries.shape[2]))
        test_X = testLSATimeSeries.reshape((testLSATimeSeries.shape[0], numChunk, testLSATimeSeries.shape[2]))
        #print(train_X.shape, Ytrain.shape, test_X.shape, Ytest.shape)
        applyingLSTM(train_X,Ytrain,test_X,Ytest)


def approachtimeSeriesLdaTfIdfLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks): 
    for numChunk in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",numChunk)
        trainTimeSeries = extractTimeSeries(trainSubjects,numChunk) 
        testTimeSeries = extractTimeSeries(testSubjects,numChunk) 
        Ytrain = extractingLabels(trainSubjects,trainLabels)
        Ytest = extractingLabels(testSubjects,testLabels)
        Ytrain = np.where(Ytrain==1, 1, 0)
        Ytest = np.where(Ytest==1, 1, 0)
        
        trainLDATimeSeries , testLDATimeSeries = processTfIdfLDATimeSeries(trainTimeSeries,testTimeSeries,Ytrain)
        # reshape input to be 3D [samples, timesteps, features]
        train_X = trainLDATimeSeries.reshape((trainLDATimeSeries.shape[0], numChunk, trainLDATimeSeries.shape[2]))
        test_X = testLDATimeSeries.reshape((testLDATimeSeries.shape[0], numChunk, testLDATimeSeries.shape[2]))
        #print(train_X.shape, Ytrain.shape, test_X.shape, Ytest.shape)
        applyingLSTM(train_X,Ytrain,test_X,Ytest)

def approachtimeSeriesDocToVecLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks): 
    for numChunk in range (1,nbChunks+1) : 
        print("\n\nCHUNK ",numChunk)
        trainTimeSeries = extractTimeSeries(trainSubjects,numChunk) 
        testTimeSeries = extractTimeSeries(testSubjects,numChunk) 
        Ytrain = extractingLabels(trainSubjects,trainLabels)
        Ytest = extractingLabels(testSubjects,testLabels)
        Ytrain = np.where(Ytrain==1, 1, 0)
        Ytest = np.where(Ytest==1, 1, 0)
        trainDTVTimeSeries , testDTVTimeSeries = processDocToVecTimeSeries(trainTimeSeries,testTimeSeries)
        
        # reshape input to be 3D [samples, timesteps, features]
        train_X = trainDTVTimeSeries.reshape((trainDTVTimeSeries.shape[0], numChunk, trainDTVTimeSeries.shape[2]))
        test_X = testDTVTimeSeries.reshape((testDTVTimeSeries.shape[0], numChunk, testDTVTimeSeries.shape[2]))
        #print(train_X.shape, Ytrain.shape, test_X.shape, Ytest.shape)
        applyingLSTM(train_X,Ytrain,test_X,Ytest)
