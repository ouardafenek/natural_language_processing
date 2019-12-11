from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from matplotlib import pyplot as plt

from processor import * 
from evaluator import * 


def classify( Xtrain, Ytrain, Xtest, Ytest): 
    #applique les classifieurs sur les données passées en paramètres, 
    #ces données sont supposées etre dan sle bon format 
    
    #classifieurs à appliquer 
    #print("Les classifieus appliqués sont : ")
    #print("LINEAR SVM")
    #print("LOGISTIC REGRESSION")
    #print("RANDOM FOREST")
    #print("MLP")
    
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    svm = LinearSVC()
    logreg = LogisticRegression(n_jobs=1, C=1e5)
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
    
    svm.fit(Xtrain, Ytrain)  
    y_pred = svm.predict(Xtest)  
    afficheScore(Ytest, y_pred)
    
    logreg.fit(Xtrain, Ytrain)  
    y_pred = logreg.predict(Xtest) 
    afficheScore(Ytest, y_pred)
    
    rf.fit(Xtrain, Ytrain)  
    y_pred = rf.predict(Xtest)  
    afficheScore(Ytest, y_pred)
    
    cnn.fit(Xtrain, Ytrain)  
    y_pred = cnn.predict(Xtest) 
    afficheScore(Ytest, y_pred)


def applyingModels ( Xtrain, Ytrain, Xtest, Ytest): 
    
    
    #Partie TF-IDF 
    #train_corpus_tf_idf, test_corpus_tf_idf = processTFIDF(Xtrain, Xtest)
    #classify( train_corpus_tf_idf, Ytrain, test_corpus_tf_idf, Ytest)
    
    ################################################
    #Partie TF-IDF et LSA 
    #train_corpus_tf_idf_lsa, test_corpus_tf_idf_lsa = processLSA(train_corpus_tf_idf, test_corpus_tf_idf)
    #classify( train_corpus_tf_idf_lsa, Ytrain, test_corpus_tf_idf_lsa, Ytest)
    
    ################################################
    #Partie Words Embeddings (Doc To Vec)

    train_corpus_doc2vec ,test_corpus_doc2vec = processDocToVec(Xtrain, Xtest)
    classify( train_corpus_doc2vec, Ytrain, test_corpus_doc2vec, Ytest)
    
    ################################################
    #Partie LDA combiné avec TF-IDF 
    
    #train_corpus_tf_idf_lda, test_corpus_tf_idf_lda = processLDA(train_corpus_tf_idf,test_corpus_tf_idf,Ytrain)
    #classify( train_corpus_tf_idf_lda, Ytrain, test_corpus_tf_idf_lda, Ytest)
    
################################################################################################################
############################### FONCTIONS POUR LES SERIES TEMPORELLES       ####################################
################################################################################################################
def applyingLSTM(train_X,Ytrain,test_X,Ytest): 
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    #model.compile(loss='mae', optimizer='adam')
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    #print(model.summary())
    # fit network
    history = model.fit(train_X, Ytrain, epochs=35, batch_size=72, validation_data=(test_X, Ytest), verbose=2, shuffle=False)
    # make a prediction
    yhat = model.predict(test_X)
    yy = np.where(yhat>0.5, 1 , 0)
    afficheScore(Ytest, yy)

    # plot history
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()


    
################################################################################################################
############################### FONCTIONS POUR LA PARTIE PREDICTION PRECOCE ####################################
################################################################################################################

####Pour les quatres fonctions suivantes, la seule différence est le classifieur et/ou la méthode de reprrésentation 
####sinon, l'idée est la même dans les quatres. 

def applyingRegLogLsaTfIdfThreshold ( Xtrain, Ytrain, Xtest, Ytest, numChunk, numPredChunk, lastPred): 

    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    #numChunk est le numéro du chunk en cours 
    #lastpred : tableau de prédictions du chunk d'avant 
    
    train_corpus_tf_idf, test_corpus_tf_idf = processTFIDF(Xtrain, Xtest)
    train_corpus_lsa, test_corpus_lsa = processLSA(train_corpus_tf_idf, test_corpus_tf_idf)
    
    logreg_lsa = LogisticRegression(n_jobs=1, C=1e5)
    logreg_lsa.fit(train_corpus_lsa, Ytrain)
    
    y_pred = logreg_lsa.predict(test_corpus_lsa)
    y_pred_proba = logreg_lsa.predict_proba(test_corpus_lsa)
    #probas d'appartenance à la classe majoritaire
    y_pred_proba = (np.max(y_pred_proba,axis=1))
    #Formule du seuil : moyenne des probas de la classe majoritaire - variance 
    thresholdPb = np.mean(y_pred_proba) - np.var(y_pred_proba)
    #Si on est au dernier chunk, on met le seuil à 0, 
    #pour etre sur qu'on va emmettre une décision pour tous les utilisateurs
    if (numChunk==10): 
        thresholdPb=0
        
    resultat = []
    for decision,proba in zip (y_pred,y_pred_proba):
        if(proba > thresholdPb):
            #on va emmettre une décision qui sera 1 ou -1
            resultat.append(decision)     
        else: 
            #On va emmettre un 0, cela voudrait dire qu'on a pas encore pris de décision 
            resultat.append(0.0)
   
    resultat = np.asarray(resultat)

    #Les gens pour lesquels on a déja émis une décision positive (anorexique), 
    #on cosidère la décision déjà émise, elle ne peut pas changer 
    for i in range(len(Ytest)): 
        if lastPred[i]== 1 : #Si la personne a été clssée comme anorexique, elle le reste 
            resultat[i] = lastPred[i]
            
        elif resultat[i]!=0 : 
            #si on a pas déjà prédit que i est anorexique, et qu'on vient d'attribuer 
            #une décision pour i alors on est bien dans le chunk de prédiction pour i 
            #cette condition permet que si par exp au chunk 2 on prédit -1 pour i 
            #et au 5 on prédit 1, alors le chunk de préd sera 5 (c'est ce qu'on veut) 
            numPredChunk[i] = numChunk 
        
    #on sauvegarge le résultat pour la prochaine itération 
    lastPred = resultat     
   
    #Evaluation : Precision, Recall et F1 
    #on fera attention à ne prendre que les lignes pour lequelles on a émis une décision
    
    indicesConcernes = np.where(resultat!=0.0)
    Ytest = Ytest[indicesConcernes]
    y_pred = resultat[indicesConcernes]
    
    afficheScore(Ytest, y_pred)
    return lastPred, numPredChunk, y_pred 




def applyingCnnDocToVecThreshold ( Xtrain, Ytrain, Xtest, Ytest, numChunk, numPredChunk, lastPred): 
    
    train_corpus_doctovec, test_corpus_doctovec = processDocToVec(Xtrain, Xtest)
    
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
    cnn.fit(train_corpus_doctovec, Ytrain)
    y_pred = cnn.predict(test_corpus_doctovec)
    y_pred_proba = cnn.predict_proba(test_corpus_doctovec)
    y_pred_proba = (np.max(y_pred_proba,axis=1))
    
    
    thresholdPb = np.mean(y_pred_proba) - np.var(y_pred_proba)
    #si on est au dixième chunk, on met le seuil à 0 
    #c'est à dire dans ce cas, on oblige le modèle à y mettre un décision 
    if (numChunk == 10) :
        thresholdPb=0 
    #print(y_pred_proba)
    
    resultat = []
    for decision,proba in zip (y_pred,y_pred_proba):
        if(proba > thresholdPb):
            #on va emmettre une décision qui sera 1 ou -1, attention, il faut faire attention pour 
            #savoir à quelle sujet cette décision correspond t elle 
            resultat.append(decision)     
        else: 
            #On va emmettre un 0, cela voudrait dire qu'on a pas encore pris de décision 
            resultat.append(0.0)
   
    resultat = np.asarray(resultat)

    #Les gens pour lesquels on a déja émis une décision positive (anorexiques), 
    #on cosidère la décision déjà émise, elle ne peut pas changer 
    for i in range(len(Ytest)): 
        if lastPred[i]== 1 : #Si la personne a été clssée comme anorexique, elle le reste 
            resultat[i] = lastPred[i]
            
        elif resultat[i]!=0 : 
            #si on a pas déjà prédit que i est anorexique, et qu'on vient d'attribuer 
            #une décision pour i alors on est bien dans le chunk de prédiction pour i 
            #cette condition permet que si par exp au chunk 2 on prédit -1 pour i 
            #et au 5 on prédit 1, alors le chunk de préd sera 5 (c'est ce qu'on veut) 
            numPredChunk[i] = numChunk 
        
    #on sauvegarge le résultat pour la prochaine itération 
    lastPred = resultat     
    #pour cette partie on va défénir de nouvelles formules de calcul de score , 
    #on ne va plus comparer Ytest avec y_pred mais plutot Ytest avec resultat 
    #on fera attention lors du calcul de la précision et du recall à ne prendre que les lignes pour 
    #lequelles on a déjà émis une décision, 
    indicesConcernes = np.where(resultat!=0.0)
    Ytest = Ytest[indicesConcernes]
    y_pred = resultat[indicesConcernes]
    afficheScore(Ytest, y_pred)
    return lastPred, numPredChunk, y_pred


def applyingCnnLsaTfIdfThreshold ( Xtrain, Ytrain, Xtest, Ytest, numChunk, numPredChunk, lastPred): 

    #pour ERDE: il faudrait savoir à quel moment on a émis la décision c'est à ça que ça sert 
    #le tableau numPredChunk, le ERDE se calcule après le dernier chunk 
    #numChunk est le numéro du chunk en cours 
    #lastpred : tableau de prédictions du chunk d'avant 
    
    train_corpus_tf_idf, test_corpus_tf_idf = processTFIDF(Xtrain, Xtest)
    train_corpus_lsa, test_corpus_lsa = processLSA(train_corpus_tf_idf, test_corpus_tf_idf)
    
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=1)
    cnn.fit(train_corpus_lsa, Ytrain)
    
   
    
    y_pred = cnn.predict(test_corpus_lsa)
    y_pred_proba = cnn.predict_proba(test_corpus_lsa)
    #probas d'appartenance à la classe majoritaire
    y_pred_proba = (np.max(y_pred_proba,axis=1))
    #Formule du seuil : moyenne des probas de la classe majoritaire - variance 
    thresholdPb = np.mean(y_pred_proba) - np.var(y_pred_proba)
    #Si on est au dernier chunk, on met le seuil à 0, 
    #pour etre sur qu'on va emmettre une décision pour tous les utilisateurs
    if (numChunk==10): 
        thresholdPb=0
        
    resultat = []
    for decision,proba in zip (y_pred,y_pred_proba):
        if(proba > thresholdPb):
            #on va emmettre une décision qui sera 1 ou -1
            resultat.append(decision)     
        else: 
            #On va emmettre un 0, cela voudrait dire qu'on a pas encore pris de décision 
            resultat.append(0.0)
   
    resultat = np.asarray(resultat)

    #Les gens pour lesquels on a déja émis une décision positive (anorexique), 
    #on cosidère la décision déjà émise, elle ne peut pas changer 
    for i in range(len(Ytest)): 
        if lastPred[i]== 1 : #Si la personne a été clssée comme anorexique, elle le reste 
            resultat[i] = lastPred[i]
            
        elif resultat[i]!=0 : 
            #si on a pas déjà prédit que i est anorexique, et qu'on vient d'attribuer 
            #une décision pour i alors on est bien dans le chunk de prédiction pour i 
            #cette condition permet que si par exp au chunk 2 on prédit -1 pour i 
            #et au 5 on prédit 1, alors le chunk de préd sera 5 (c'est ce qu'on veut) 
            numPredChunk[i] = numChunk 
        
    #on sauvegarge le résultat pour la prochaine itération 
    lastPred = resultat     
   
    #Evaluation : Precision, Recall et F1 
    #on fera attention à ne prendre que les lignes pour lequelles on a émis une décision
    
    indicesConcernes = np.where(resultat!=0.0)
    Ytest = Ytest[indicesConcernes]
    y_pred = resultat[indicesConcernes]
    
    afficheScore(Ytest, y_pred)
    return lastPred, numPredChunk, y_pred 


def applyingRegLogDocToVecThreshold ( Xtrain, Ytrain, Xtest, Ytest, numChunk, numPredChunk, lastPred): 
    
    train_corpus_doctovec, test_corpus_doctovec = processDocToVec(Xtrain, Xtest)
    
    logreg_lsa = LogisticRegression(n_jobs=1, C=1e5)
    logreg_lsa.fit(train_corpus_doctovec, Ytrain)
    y_pred = logreg_lsa.predict(test_corpus_doctovec)
    y_pred_proba = logreg_lsa.predict_proba(test_corpus_doctovec)
    y_pred_proba = (np.max(y_pred_proba,axis=1))
    
    
    thresholdPb = np.mean(y_pred_proba) - np.var(y_pred_proba)
    #si on est au dixième chunk, on met le seuil à 0 
    #c'est à dire dans ce cas, on oblige le modèle à y mettre un décision 
    if (numChunk == 10) :
        thresholdPb=0 
    #print(y_pred_proba)
    
    resultat = []
    for decision,proba in zip (y_pred,y_pred_proba):
        if(proba > thresholdPb):
            #on va emmettre une décision qui sera 1 ou -1, attention, il faut faire attention pour 
            #savoir à quelle sujet cette décision correspond t elle 
            resultat.append(decision)     
        else: 
            #On va emmettre un 0, cela voudrait dire qu'on a pas encore pris de décision 
            resultat.append(0.0)
   
    resultat = np.asarray(resultat)

    #Les gens pour lesquels on a déja émis une décision positive (anorexiques), 
    #on cosidère la décision déjà émise, elle ne peut pas changer 
    for i in range(len(Ytest)): 
        if lastPred[i]== 1 : #Si la personne a été clssée comme anorexique, elle le reste 
            resultat[i] = lastPred[i]
            
        elif resultat[i]!=0 : 
            #si on a pas déjà prédit que i est anorexique, et qu'on vient d'attribuer 
            #une décision pour i alors on est bien dans le chunk de prédiction pour i 
            #cette condition permet que si par exp au chunk 2 on prédit -1 pour i 
            #et au 5 on prédit 1, alors le chunk de préd sera 5 (c'est ce qu'on veut) 
            numPredChunk[i] = numChunk 
        
    #on sauvegarge le résultat pour la prochaine itération 
    lastPred = resultat     
    #pour cette partie on va défénir de nouvelles formules de calcul de score , 
    #on ne va plus comparer Ytest avec y_pred mais plutot Ytest avec resultat 
    #on fera attention lors du calcul de la précision et du recall à ne prendre que les lignes pour 
    #lequelles on a déjà émis une décision, 
    indicesConcernes = np.where(resultat!=0.0)
    Ytest = Ytest[indicesConcernes]
    y_pred = resultat[indicesConcernes]
    afficheScore(Ytest, y_pred)
    return lastPred, numPredChunk, y_pred
