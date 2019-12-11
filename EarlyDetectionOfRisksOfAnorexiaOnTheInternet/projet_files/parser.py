from lxml import etree
import os
import numpy as np

def extractId(filename):
    return filename.split("_")[0][7:]
def extractIdGoldenTruthTest(subjectname):
    return subjectname[7:]

def labelingTrain (path):
    #Etant donné l'emplacement des données d'entrainement,
    #qui d'après le description seraient dans deux repertoires : negative_examples et positive_examples
    #la fonction crée un vecteur de labels pour chaque sujet et ce, celon le repertoire ou il se trouve
    #Les sujets se trouvant dans positive_examples sont labelisés avec 1 (considérés comme Anorexiques)
    #les autres avec -1 
    labels={}
    negativeExamplesPath = path+"/negative_examples/chunk1" 
    #Pour le chunk on peut prendre n'importe quel numéro, il sert juste pour retrouver les id des sujets
    for filename in os.listdir(negativeExamplesPath):
        extension=os.path.splitext(filename)
        if ".xml" in extension[1]:
            labels[extractId(filename)]= -1 
    positiveExamplesPath = path+"/positive_examples/chunk1" 
    #Pour le chunk on peut prendre n'importe quel numéro, il sert juste pour retrouver les id des sujets
    for filename in os.listdir(positiveExamplesPath):
        extension=os.path.splitext(filename)
        if ".xml" in extension[1]:
            labels[extractId(filename)]= 1
    return labels

def labelingTest(path):
    #Etant donné l'emplacement des données de test, on sait, d'après la desription que le repertoire les 
    #contenant, contient aussi un fichier nommé risk-golden-truth-test.txt qui représente les labels des sujets
    filename = path+"/risk-golden-truth-test.txt"
    labels={}
    f = open(filename,'r')
    lines = f.readlines()
    for line in lines : 
            subject,decision = line.split()
            if(decision=="0"):
                labels[extractIdGoldenTruthTest(subject)]=-1
            else:
                labels[extractIdGoldenTruthTest(subject)]=1
    return labels

def parsing (path, features,nbChunks):
    #paramètres: 
    #path : chemin vers les chunks
    #features: ce qu'on extrait de chaque document 
    #nbChuns: Le nombre total de chunks dont on dispose   
    #Le retour: 
    #Un dictionnaire dont les clés sont les identifiants des sujets 
    #ses valeurs, sont aussi des dictionnaires pour référer les documents (les chunks)
    #et enfin pour chaque chunk, on a un dictionnaire des features
    
    nbChunks +=1 #Manipulation, parce qu'on commence de 1 et non pas de 0 
    subjects = {}
    for i in range(1,nbChunks):
        #On va explorer le chunk i pour les deux categiries (negative and positive examples) dans le cas 
        #des exemples du train 
        if(path[-5:]=='train'):
            chunkPaths = [path+"/negative_examples/chunk"+str(i), path+"/positive_examples/chunk"+str(i)]
        else:
            chunkPaths = [path+"/chunk"+str(i)]
            
        for chunkPath in chunkPaths: 
            #For all the subjects in the repository 
            for filename in os.listdir(chunkPath):
                #On va tester que le fichier à lire est bien au format XML
                extension=os.path.splitext(filename)
                if ".xml" in extension[1]:
                    #Reading the XML file
                    tree = etree.parse(chunkPath+"/"+filename)
                    #Extracting the id from the filename
                    idSubject = extractId(filename)
                    #Initialisation des dictionnaires pour chaque sujet
                    if i==1 : 
                        #On crée le dictionnaire des chunks si c'est le premier chunk 
                        subjects[idSubject] = {i:dict() for i in range(1,nbChunks)}
                    
                    #initialisation du dictionnaire pour chaque chunk
                    subjects[idSubject][i] = {f:dict() for f in features}
                    
                    #On fera une boucle, qui pour chaque sujet, 
                    #renvoie tous les textes publiés dans une partie (dans un chunk donné)
                    #On concatène ces texte là pour former un document 
                    writings = tree.xpath("/INDIVIDUAL/WRITING")
                    subjects[idSubject][i]['text'] = {i:dict() for i in range(len(writings))}
                    for j,writing in enumerate (writings): 
                        text = writing.getchildren()[3]
                        title = writing
                        subjects[idSubject][i]['text'][j] =  text.text

                    #Au cas l'utlisateur n'a rien publié, on fera cette manipulation pour éviter la dévision par 0
                    #dans les calculs des moyennes après
                    nbWritings = len(tree.xpath("/INDIVIDUAL/WRITING"))
                    if (nbWritings==0):
                        nbWritings =1 

    return subjects



def extractingChunkText (trainSubjects, trainLabels, testSubjects,testLabels,numChunk):
	#extraire le texte ecrit dans le chunk "numChunk" pour chaque utilisateur 
	#ce texte est alors la concaténation de tout les writings (soumissions) de cet utilisateur dans ce chunk 
	#ici, on enlève les id des utilisateurs on retourne des tableaux 
    Xtrain =[]
    Xtest= []
    Ytrain = np.zeros(len(trainLabels))
    Ytest = np.zeros(len(testLabels))
    for i,k in enumerate(trainSubjects.keys()) :
        Ytrain[i] = trainLabels[k]
        text = ""
        for key in trainSubjects[k][numChunk]['text'].keys(): 
            text += trainSubjects[k][numChunk]['text'][key]
        Xtrain.append(text)
        
    for i,k in enumerate(testSubjects.keys()) :
        Ytest[i] = testLabels[k]
        text = ""
        for key in testSubjects[k][numChunk]['text'].keys(): 
            text += testSubjects[k][numChunk]['text'][key]
        Xtest.append(text)
    return Xtrain, Ytrain, Xtest, Ytest

def extractingNChunkText (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks):
    Xtrain =[]
    Xtest= []
    Ytrain = np.zeros(len(trainLabels))
    Ytest = np.zeros(len(testLabels))
     
    for i,k in enumerate(trainSubjects.keys()) :
        text = ""
        Ytrain[i] = trainLabels[k]
        for j in range (1,nbChunks+1):
            for key in trainSubjects[k][j]['text'].keys(): 
                text += trainSubjects[k][j]['text'][key]
        Xtrain.append(text)
    for i,k in enumerate(testSubjects.keys()) :
        text = ""
        Ytest[i] = testLabels[k]
        for j in range (1,nbChunks+1):
            for key in testSubjects[k][j]['text'].keys(): 
                text += testSubjects[k][j]['text'][key]
        Xtest.append(text)
    return Xtrain, Ytrain, Xtest, Ytest


#Dans ce qui suit nous allons considérer les soumissions indépendemment pas par chunk 
def extractingWritingsFromChunk(trainSubjects, trainLabels, testSubjects,testLabels,numChunk):
    Xtrain =[]
    Xtest= []
    Ytrain = []
    Ytest = []
    for subject in trainSubjects.keys() :
        for writing in trainSubjects[subject][numChunk]['text'].keys():
            Ytrain.append(trainLabels[subject])
            Xtrain.append(trainSubjects[subject][numChunk]['text'][writing])
    for subject in testSubjects.keys() :
        for writing in testSubjects[subject][numChunk]['text'].keys():
            Ytest.append(testLabels[subject])
            Xtest.append(testSubjects[subject][numChunk]['text'][writing])
        
    return np.array(Xtrain),np.array(Ytrain),np.array(Xtest),np.array(Ytest)

def extractingWritingsFromNChunk(trainSubjects, trainLabels, testSubjects,testLabels,nbChunk):
    Xtrain =[]
    Xtest= []
    Ytrain = []
    Ytest = []
    for numChunk in range(1,nbChunk+1):
        for subject in trainSubjects.keys() :
            for writing in trainSubjects[subject][numChunk]['text'].keys():
                Ytrain.append(trainLabels[subject])
                Xtrain.append(trainSubjects[subject][numChunk]['text'][writing])
        for subject in testSubjects.keys() :
            for writing in testSubjects[subject][numChunk]['text'].keys():
                Ytest.append(testLabels[subject])
                Xtest.append(testSubjects[subject][numChunk]['text'][writing])

    return np.array(Xtrain),np.array(Ytrain),np.array(Xtest),np.array(Ytest)



#on va modifier la fonction extractingNChunkText pour qu'elle renvoie aussi 
#le nombre de writing de chaque sujet de test dans chaque chunk (on en aura besoin pour ERDE)
#On s'interesse pas au nombre de ceux du train (puisqu'on ne va pas scorer ceux la)
def extractingNChunkText2 (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks):
    Xtrain =[]
    Xtest= []
    Ytrain = np.zeros(len(trainLabels))
    Ytest = np.zeros(len(testLabels))
    for i,k in enumerate(trainSubjects.keys()) :
        text = ""
        Ytrain[i] = trainLabels[k]
        for j in range (1,nbChunks+1):
            for key in trainSubjects[k][j]['text'].keys(): 
                text += trainSubjects[k][j]['text'][key]
            
        Xtrain.append(text)
        
    nbWritings = np.zeros((len(testSubjects.keys()), nbChunks))
    for i,k in enumerate(testSubjects.keys()) :
        text = ""
        Ytest[i] = testLabels[k]
        for j in range (1,nbChunks+1):
            nbWritings[i][j-1] = len(testSubjects[k][j]['text'].keys())
            for key in testSubjects[k][j]['text'].keys(): 
                text += testSubjects[k][j]['text'][key]
        Xtest.append(text)
    return Xtrain, Ytrain, Xtest, Ytest, nbWritings


#méthodes d'extraction de texte de façon adéquate pour les séries temporelles 
def extractChunkText(subjects,numChunk):
    X =[]
    for i,k in enumerate(subjects.keys()) :
        text = ""
        for key in subjects[k][numChunk]['text'].keys(): 
            text += subjects[k][numChunk]['text'][key]
        X.append(text)
    return np.array(X) 
def extractingLabels(subjects,labels): 
	#On enlève les clés, pour retourner un numpy array 
    Y=[]
    for k in subjects.keys() :
        Y.append(labels[k])
    return np.array(Y)
def extractTimeSeries(subjects,nbChunks) : 
    X=[]
    for numChunk in range(1,nbChunks+1): 
        X.append(extractChunkText(subjects,numChunk))
    X=np.array(X)
    #comme on avait extrait chunk par chunk, on retourne la transposée, 
    #comme ça pour chaque utilisateur on a une série de 10 textes qu'il a écrit sur 10 pas de temps
    return X.T
