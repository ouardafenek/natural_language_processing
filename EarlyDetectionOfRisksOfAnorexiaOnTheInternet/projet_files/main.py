import sys
from parser import * 
from approaches import * 

trainPath = "../trainingData/train"
testPath = "../trainingData/test"
features = ['text']
nbChunks = 10 
if __name__ =='__main__': 
	trainSubjects = parsing (trainPath, features,nbChunks)
	trainLabels = labelingTrain(trainPath)
	testSubjects = parsing (testPath, features,nbChunks)
	testLabels = labelingTest(testPath)
	approach1(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 
	#approach2(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 
	#approach3(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 
	#approach4(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 

	#approachtimeSeriesLsaTfIdfLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks)
	#approachtimeSeriesLdaTfIdfLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks)
	#approachtimeSeriesDocToVecLSTM (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks)


	#approach2TresholdRegLogLsaTfIdf(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 
	#approach2TresholdRegLogDocToVec(trainSubjects, trainLabels, testSubjects,testLabels,nbChunks) 
	#approach2TresholdCnnLsaTfIdf (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks)
	#approach2TresholdCnnDocToVec (trainSubjects, trainLabels, testSubjects,testLabels,nbChunks)


