import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from sklearn.utils import shuffle

######################################################################
######################################################################
######################################################################
## CSP File from: https://github.com/spolsley/common-spatial-patterns

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(*tasks):
	if len(tasks) < 2:
		print("Must have at least 2 tasks for filtering.")
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)
######################################################################
def GetMinSteps(indeces, data):
	minVal = 9999;
	for index in indeces:
		length = data[index].shape[1];
		if(length < minVal):
			minVal = length;
	return minVal
######################################################################
def GetData(indeces, dataIn, truncateValue):
	dataOut = []
	truncate = True;
	if truncateValue == 0:
		truncate = False;
	for idx in indeces:
		currentData = dataIn[idx]
		if truncate:
			dataOut.append(currentData[:,0:truncateValue])
		else:
			dataOut.append(currentData)
	return np.asarray(dataOut)
######################################################################
######################################################################
######################################################################

def GetCombinedData_5F(relativeFilePath, shuffleData):
	fileNames = ['5F-SubjectB-160309-5St-SGLHand-HFREQ.mat','5F-SubjectB-160311-5St-SGLHand-HFREQ.mat',\
             '5F-SubjectC-160429-5St-SGLHand-HFREQ.mat','5F-SubjectE-160321-5St-SGLHand-HFREQ.mat',\
			 '5F-SubjectF-160210-5St-SGLHand-HFREQ.mat','5F-SubjectG-160413-5St-SGLHand-HFREQ.mat',\
			 '5F-SubjectG-160428-5St-SGLHand-HFREQ.mat','5F-SubjectH-160804-5St-SGLHand-HFREQ.mat',\
			 '5F-SubjectI-160719-5St-SGLHand-HFREQ.mat','5F-SubjectI-160723-5St-SGLHand-HFREQ.mat'];
	numDatasets = len(fileNames);
	for dataset in range(0, numDatasets):
		print("Processing dataset {} of {}".format(dataset+1, numDatasets))

		fileName = fileNames[dataset];

		file = sio.loadmat(relativeFilePath+'/{}'.format(fileName)) #replace with .mat file name
		header=file['__header__']
		version=file['__version__']
		glob=file['__globals__']
		#ans=file['ans']


		#x=file['x']
		o=file['o'][0][0]
		data=o['data']
		data = np.transpose(data)
		data = data[0:21,:];
		#print(data)
		nS=o['nS'][0][0]
		#values of structure seem to be 2D numpy arrays, if originally a scalar in Matlab.
		#use [0][0] to get scalar.
		#print("Number of samples: {numSamples}".format(numSamples=nS))
		test=o['id'][0] #id value became a 1D array of size 1 for some reason. use [0] to get value
		#print("Dataset ID: {id}".format(id=test))
		chnames=o['chnames'][:,0] #[:,0] converts from 2D array back to 1D array
		#print("Channel names: {channelNames}".format(channelNames=chnames))
		markers = o['marker']
		## The markers are all still individual arrays of size 1x1, so we convert them to an array with single values
		markersArray = []
		for marker in markers:
			markersArray.append(marker[0])
		markersArray = np.asarray(markersArray)

		## Find the starting indeces where the marker changes
		changeIdxs = np.where(np.transpose(markersArray)[:-1] != np.transpose(markersArray)[1:])[0]
		#print("Number of index changes: {idxChanges}".format(idxChanges=changeIdxs.shape[0]))
		## Split the data so that it has its matching marker
		dataSplit = np.array_split(data, changeIdxs[:-1], axis=1)
		splitCount = 0
		for splitData in dataSplit:
			splitCount += 1
		#print("Number of arrays in data split: {num}".format(num=splitCount))
		## Retrieve the marker values for each of the change indeces (changeIdxs)
		markerTargets = markersArray[changeIdxs];
		#print("Number of marker targets: {numTargets}".format(numTargets=markerTargets.shape[0]))

		## To Apply CSP, we first only get the indeces for MI tasks 1 and 2 (left and right hand, respectively.)
		tIdx = np.where(markerTargets == 1)[0]
		iIdx = np.where(markerTargets == 2)[0]
		mIdx = np.where(markerTargets == 3)[0]
		rIdx = np.where(markerTargets == 4)[0]
		pIdx = np.where(markerTargets == 5)[0]

		tIdxMin = GetMinSteps(tIdx, dataSplit)
		iIdxMin = GetMinSteps(iIdx, dataSplit)
		mIdxMin = GetMinSteps(mIdx, dataSplit)
		rIdxMin = GetMinSteps(rIdx, dataSplit)
		pIdxMin = GetMinSteps(pIdx, dataSplit)
		minValues = [tIdxMin, iIdxMin, mIdxMin, rIdxMin, pIdxMin]
		#minValues

		#Truncate the data to the min size
		#print(minValue)

		if(dataset == 0):
			
			minValue = np.min(minValues)
			
			tData = GetData(tIdx, dataSplit, minValue)
			iData = GetData(iIdx, dataSplit, minValue)
			mData = GetData(mIdx, dataSplit, minValue)
			rData = GetData(rIdx, dataSplit, minValue)
			pData = GetData(pIdx, dataSplit, minValue)

			#pauseStr = input("Pausing after iteration: " + str(dataset))
			minLen = np.min([len(tData), len(iData), len(mData), len(rData), len(pData)])
			
			tData = tData[0:minLen]
			iData = iData[0:minLen]
			mData = mData[0:minLen]
			rData = rData[0:minLen]
			pData = pData[0:minLen]
			
			#Construct the target arrays and merge the data
			#tTargets = np.tile(np.array([1,0,0,0,0]),(minLen,1))
			#iTargets = np.tile(np.array([0,1,0,0,0]),(minLen,1))
			#mTargets = np.tile(np.array([0,0,1,0,0]),(minLen,1))
			#rTargets = np.tile(np.array([0,0,0,1,0]),(minLen,1))
			#pTargets = np.tile(np.array([0,0,0,0,1]),(minLen,1))
			
			
		else:
			tempMinValue = np.min(minValues)
			temptData = GetData(tIdx, dataSplit, tempMinValue)
			tempiData = GetData(iIdx, dataSplit, tempMinValue)
			tempmData = GetData(mIdx, dataSplit, tempMinValue)
			temprData = GetData(rIdx, dataSplit, tempMinValue)
			temppData = GetData(pIdx, dataSplit, tempMinValue)
			
			tempminLen = np.min([len(temptData), len(tempiData), len(tempmData), len(temprData), len(temppData)])
			temptData = temptData[0:minLen]
			tempiData = tempiData[0:minLen]
			tempmData = tempmData[0:minLen]
			temprData = temprData[0:minLen]
			temppData = temppData[0:minLen]
			
			#Construct the target arrays and merge the data
			#temptTargets = np.tile(np.array([1,0,0,0,0]),(tempminLen,1))
			#tempiTargets = np.tile(np.array([0,1,0,0,0]),(tempminLen,1))
			#tempmTargets = np.tile(np.array([0,0,1,0,0]),(tempminLen,1))
			#temprTargets = np.tile(np.array([0,0,0,1,0]),(tempminLen,1))
			#temppTargets = np.tile(np.array([0,0,0,0,1]),(tempminLen,1))
			
			#tTargets = np.vstack((tTargets, temptTargets));
			#iTargets = np.vstack((iTargets, tempiTargets));
			#mTargets = np.vstack((mTargets, tempmTargets));
			#rTargets = np.vstack((rTargets, temprTargets));
			#pTargets = np.vstack((pTargets, temppTargets));
			
			## We have to check if the new minValue is less than the previous minValue
			if(tempMinValue < minValue):
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#tDelete = np.delete(tData, 1275, 1);
				#print(tDelete.shape)
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				#In this case, we have to truncate the values already stored to the current minValue
				tData = np.vstack((tData[:, :, 0:tempMinValue], temptData));
				iData = np.vstack((iData[:, :, 0:tempMinValue], tempiData));
				mData = np.vstack((mData[:, :, 0:tempMinValue], tempmData));
				rData = np.vstack((rData[:, :, 0:tempMinValue], temprData));
				pData = np.vstack((pData[:, :, 0:tempMinValue], temppData));
				minValue = tempMinValue;
			elif(tempMinValue == minValue):
				#In this case, we don't need truncation
				tData = np.vstack((tData, temptData));
				iData = np.vstack((iData, tempiData));
				mData = np.vstack((mData, tempmData));
				rData = np.vstack((rData, temprData));
				pData = np.vstack((pData, temppData));
			else:
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				tData = np.vstack((tData, temptData[:, :, 0:minValue]));
				iData = np.vstack((iData, tempiData[:, :, 0:minValue]));
				mData = np.vstack((mData, tempmData[:, :, 0:minValue]));
				rData = np.vstack((rData, temprData[:, :, 0:minValue]));
				pData = np.vstack((pData, temppData[:, :, 0:minValue]));
                
    

	##Combine the data by stacking the individual markers and finger data
	lentData = len(tData);
	leniData = len(iData);
	lenmData = len(mData);
	lenrData = len(rData);
	lenpData = len(pData);

	tTargets = np.tile(np.array([1,0,0,0,0]),(lentData,1))
	iTargets = np.tile(np.array([0,1,0,0,0]),(leniData,1))
	mTargets = np.tile(np.array([0,0,1,0,0]),(lenmData,1))
	rTargets = np.tile(np.array([0,0,0,1,0]),(lenrData,1))
	pTargets = np.tile(np.array([0,0,0,0,1]),(lenpData,1))

	Data = np.vstack((tData, iData, mData, rData, pData));
	Targets = np.vstack((tTargets, iTargets, mTargets, rTargets, pTargets));

	filters = CSP(tData, iData, mData, rData, pData);
	filtersArray = np.asarray(filters)
	tData = np.matmul(np.transpose(filtersArray[0]), tData);
	iData = np.matmul(np.transpose(filtersArray[1]), iData);
	mData = np.matmul(np.transpose(filtersArray[2]), mData);
	rData = np.matmul(np.transpose(filtersArray[3]), rData);
	pData = np.matmul(np.transpose(filtersArray[4]), pData);

	DataCSP = np.vstack((tData, iData, mData, rData, pData));
	TargetsCSP = np.vstack((tTargets, iTargets, mTargets, rTargets, pTargets));

	if(shuffleData == True):
		Data, Targets = shuffle(Data, Targets, random_state=0)
		DataCSP, TargetsCSP = shuffle(DataCSP, TargetsCSP, random_state=0)
	return (Data, Targets, DataCSP, TargetsCSP)


def GetCombinedData_CLA(relativeFilePath, shuffleData):
	fileNames = ['CLASubjectA1601083StLRHand.mat',\
             'CLASubjectB1510193StLRHand.mat',\
             'CLASubjectB1510203StLRHand.mat',\
             'CLASubjectB1512153StLRHand.mat',\
             'CLASubjectC1511263StLRHand.mat',\
             'CLASubjectC1512163StLRHand.mat',\
             'CLASubjectC1512233StLRHand.mat',\
             'CLASubjectD1511253StLRHand.mat',\
             'CLASubjectE1512253StLRHand.mat',\
             'CLASubjectE1601193StLRHand.mat',\
             'CLASubjectE1601223StLRHand.mat',\
             'CLASubjectF1509163StLRHand.mat',\
             'CLASubjectF1509173StLRHand.mat',\
             'CLASubjectF1509283StLRHand.mat'];
	numDatasets = len(fileNames);
	for dataset in range(0, numDatasets):
		print("Processing dataset {} of {}".format(dataset+1, numDatasets))

		fileName = fileNames[dataset];

		file = sio.loadmat(relativeFilePath+'/{}'.format(fileName)) #replace with .mat file name
		header=file['__header__']
		version=file['__version__']
		glob=file['__globals__']
		#ans=file['ans']


		#x=file['x']
		o=file['o'][0][0]
		data=o['data']
		data = np.transpose(data)
		data = data[0:21,:];
		#print(data)
		nS=o['nS'][0][0]
		#values of structure seem to be 2D numpy arrays, if originally a scalar in Matlab.
		#use [0][0] to get scalar.
		#print("Number of samples: {numSamples}".format(numSamples=nS))
		test=o['id'][0] #id value became a 1D array of size 1 for some reason. use [0] to get value
		#print("Dataset ID: {id}".format(id=test))
		chnames=o['chnames'][:,0] #[:,0] converts from 2D array back to 1D array
		#print("Channel names: {channelNames}".format(channelNames=chnames))
		markers = o['marker']
		## The markers are all still individual arrays of size 1x1, so we convert them to an array with single values
		markersArray = []
		for marker in markers:
			markersArray.append(marker[0])
		markersArray = np.asarray(markersArray)

		## Find the starting indeces where the marker changes
		changeIdxs = np.where(np.transpose(markersArray)[:-1] != np.transpose(markersArray)[1:])[0]
		#print("Number of index changes: {idxChanges}".format(idxChanges=changeIdxs.shape[0]))
		## Split the data so that it has its matching marker
		dataSplit = np.array_split(data, changeIdxs[:-1], axis=1)
		splitCount = 0
		for splitData in dataSplit:
			splitCount += 1
		#print("Number of arrays in data split: {num}".format(num=splitCount))
		## Retrieve the marker values for each of the change indeces (changeIdxs)
		markerTargets = markersArray[changeIdxs];
		#print("Number of marker targets: {numTargets}".format(numTargets=markerTargets.shape[0]))

		## To Apply CSP, we first only get the indeces for MI tasks 1 and 2 (left and right hand, respectively.)
		lhIdx = np.where(markerTargets == 1)[0]
		rhIdx = np.where(markerTargets == 2)[0]

		lhIdxMin = GetMinSteps(lhIdx, dataSplit)
		rhIdxMin = GetMinSteps(rhIdx, dataSplit)
		minValues = [lhIdxMin, rhIdxMin]
		#minValues

		#Truncate the data to the min size
		#print(minValue)

		if(dataset == 0):
			
			minValue = np.min(minValues)
			
			lhData = GetData(lhIdx, dataSplit, minValue)
			rhData = GetData(rhIdx, dataSplit, minValue)
			minLen = np.min([len(lhData), len(rhData)])
			
			lhData = lhData[0:minLen]
			rhData = rhData[0:minLen]
			
			
			
		else:
			tempMinValue = np.min(minValues)
			templhData = GetData(lhIdx, dataSplit, tempMinValue)
			temprhData = GetData(rhIdx, dataSplit, tempMinValue)

			tempminLen = np.min([len(templhData), len(temprhData)])
			templhData = templhData[0:minLen]
			temprhData = temprhData[0:minLen]
			
			## We have to check if the new minValue is less than the previous minValue
			if(tempMinValue < minValue):
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				#In this case, we have to truncate the values already stored to the current minValue
				lhData = np.vstack((lhData[:, :, 0:tempMinValue], templhData));
				rhData = np.vstack((rhData[:, :, 0:tempMinValue], temprhData));
				minValue = tempMinValue;
			elif(tempMinValue == minValue):
				#In this case, we don't need truncation
				lhData = np.vstack((lhData, templhData));
				rhData = np.vstack((rhData, temprhData));

			else:
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				lhData = np.vstack((lhData, templhData[:, :, 0:minValue]));
				rhData = np.vstack((rhData, temprhData[:, :, 0:minValue]));      

	##Combine the data by stacking the individual markers and finger data
	lenlhData = len(lhData);
	lenrhData = len(rhData);

	lhTargets = np.tile(np.array([1,0]),(lenlhData,1))
	rhTargets = np.tile(np.array([0,1]),(lenrhData,1))

	Data = np.vstack((lhData, rhData));
	Targets = np.vstack((lhTargets, rhTargets));

	filters = CSP(lhData, rhData);
	filtersArray = np.asarray(filters)

	lhData = np.matmul(np.transpose(filtersArray[0]), lhData);
	rhData = np.matmul(np.transpose(filtersArray[1]), rhData);

	DataCSP = np.vstack((lhData, rhData));
	TargetsCSP = np.vstack((lhTargets, rhTargets));

	if(shuffleData == True):
		Data, Targets = shuffle(Data, Targets, random_state=0)
		DataCSP, TargetsCSP = shuffle(DataCSP, TargetsCSP, random_state=0)
	return (Data, Targets, DataCSP, TargetsCSP)

def GetCombinedData_HaLT(relativeFilePath, shuffleData):
	fileNames = ['HaLTSubjectA1602236StLRHandLegTongue.mat',\
				'HaLTSubjectA1603086StLRHandLegTongue.mat',\
				'HaLTSubjectA1603106StLRHandLegTongue.mat',\
				'HaLTSubjectB1602186StLRHandLegTongue.mat',\
				'HaLTSubjectB1602256StLRHandLegTongue.mat',\
				'HaLTSubjectB1602296StLRHandLegTongue.mat',\
				'HaLTSubjectC1602246StLRHandLegTongue.mat',\
				'HaLTSubjectC1603026StLRHandLegTongue.mat',\
				'HaLTSubjectE1602196StLRHandLegTongue.mat',\
				'HaLTSubjectE1602266StLRHandLegTongue.mat',\
				'HaLTSubjectE1603046StLRHandLegTongue.mat',\
				'HaLTSubjectF1602026StLRHandLegTongue.mat',\
				'HaLTSubjectF1602036StLRHandLegTongue.mat',\
				'HaLTSubjectF1602046StLRHandLegTongue.mat',\
				'HaLTSubjectG1603016StLRHandLegTongue.mat',\
				'HaLTSubjectG1603226StLRHandLegTongue.mat',\
				'HaLTSubjectG1604126StLRHandLegTongue.mat',\
				'HaLTSubjectH1607206StLRHandLegTongue.mat',\
				'HaLTSubjectH1607226StLRHandLegTongue.mat',\
				'HaLTSubjectI1606096StLRHandLegTongue.mat',\
				'HaLTSubjectI1606286StLRHandLegTongue.mat',\
				'HaLTSubjectJ1611216StLRHandLegTongue.mat',\
				'HaLTSubjectK1610276StLRHandLegTongue.mat',\
				'HaLTSubjectK1611086StLRHandLegTongue.mat',\
				'HaLTSubjectL1611166StLRHandLegTongue.mat',\
				'HaLTSubjectL1612056StLRHandLegTongue.mat',\
				'HaLTSubjectM1611086StLRHandLegTongue.mat',\
				'HaLTSubjectM1611176StLRHandLegTongue.mat',\
				'HaLTSubjectM1611246StLRHandLegTongue.mat']
	numDatasets = len(fileNames);
	for dataset in range(0, numDatasets):
		print("Processing dataset {} of {}".format(dataset+1, numDatasets))

		fileName = fileNames[dataset];

		file = sio.loadmat(relativeFilePath+'/{}'.format(fileName)) #replace with .mat file name
		header=file['__header__']
		version=file['__version__']
		glob=file['__globals__']
		#ans=file['ans']


		#x=file['x']
		o=file['o'][0][0]
		data=o['data']
		data = np.transpose(data)
		data = data[0:21,:];
		#print(data)
		nS=o['nS'][0][0]
		#values of structure seem to be 2D numpy arrays, if originally a scalar in Matlab.
		#use [0][0] to get scalar.
		#print("Number of samples: {numSamples}".format(numSamples=nS))
		test=o['id'][0] #id value became a 1D array of size 1 for some reason. use [0] to get value
		#print("Dataset ID: {id}".format(id=test))
		chnames=o['chnames'][:,0] #[:,0] converts from 2D array back to 1D array
		#print("Channel names: {channelNames}".format(channelNames=chnames))
		markers = o['marker']
		## The markers are all still individual arrays of size 1x1, so we convert them to an array with single values
		markersArray = []
		for marker in markers:
			markersArray.append(marker[0])
		markersArray = np.asarray(markersArray)

		## Find the starting indeces where the marker changes
		changeIdxs = np.where(np.transpose(markersArray)[:-1] != np.transpose(markersArray)[1:])[0]
		#print("Number of index changes: {idxChanges}".format(idxChanges=changeIdxs.shape[0]))
		## Split the data so that it has its matching marker
		dataSplit = np.array_split(data, changeIdxs[:-1], axis=1)
		splitCount = 0
		for splitData in dataSplit:
			splitCount += 1
		#print("Number of arrays in data split: {num}".format(num=splitCount))
		## Retrieve the marker values for each of the change indeces (changeIdxs)
		markerTargets = markersArray[changeIdxs];
		#print("Number of marker targets: {numTargets}".format(numTargets=markerTargets.shape[0]))

		## To Apply CSP, we first only get the indeces for MI tasks 1 and 2 (left and right hand, respectively.)
		tIdx = np.where(markerTargets == 1)[0]
		iIdx = np.where(markerTargets == 2)[0]
		mIdx = np.where(markerTargets == 4)[0]
		rIdx = np.where(markerTargets == 5)[0]
		pIdx = np.where(markerTargets == 6)[0]

		tIdxMin = GetMinSteps(tIdx, dataSplit)
		iIdxMin = GetMinSteps(iIdx, dataSplit)
		mIdxMin = GetMinSteps(mIdx, dataSplit)
		rIdxMin = GetMinSteps(rIdx, dataSplit)
		pIdxMin = GetMinSteps(pIdx, dataSplit)
		minValues = [tIdxMin, iIdxMin, mIdxMin, rIdxMin, pIdxMin]
		#minValues

		#Truncate the data to the min size
		#print(minValue)

		if(dataset == 0):
			
			minValue = np.min(minValues)
			
			tData = GetData(tIdx, dataSplit, minValue)
			iData = GetData(iIdx, dataSplit, minValue)
			mData = GetData(mIdx, dataSplit, minValue)
			rData = GetData(rIdx, dataSplit, minValue)
			pData = GetData(pIdx, dataSplit, minValue)

			#pauseStr = input("Pausing after iteration: " + str(dataset))
			minLen = np.min([len(tData), len(iData), len(mData), len(rData), len(pData)])
			
			tData = tData[0:minLen]
			iData = iData[0:minLen]
			mData = mData[0:minLen]
			rData = rData[0:minLen]
			pData = pData[0:minLen]
			
			#Construct the target arrays and merge the data
			#tTargets = np.tile(np.array([1,0,0,0,0]),(minLen,1))
			#iTargets = np.tile(np.array([0,1,0,0,0]),(minLen,1))
			#mTargets = np.tile(np.array([0,0,1,0,0]),(minLen,1))
			#rTargets = np.tile(np.array([0,0,0,1,0]),(minLen,1))
			#pTargets = np.tile(np.array([0,0,0,0,1]),(minLen,1))
			
			
		else:
			tempMinValue = np.min(minValues)
			temptData = GetData(tIdx, dataSplit, tempMinValue)
			tempiData = GetData(iIdx, dataSplit, tempMinValue)
			tempmData = GetData(mIdx, dataSplit, tempMinValue)
			temprData = GetData(rIdx, dataSplit, tempMinValue)
			temppData = GetData(pIdx, dataSplit, tempMinValue)
			
			tempminLen = np.min([len(temptData), len(tempiData), len(tempmData), len(temprData), len(temppData)])
			temptData = temptData[0:minLen]
			tempiData = tempiData[0:minLen]
			tempmData = tempmData[0:minLen]
			temprData = temprData[0:minLen]
			temppData = temppData[0:minLen]
			
			#Construct the target arrays and merge the data
			#temptTargets = np.tile(np.array([1,0,0,0,0]),(tempminLen,1))
			#tempiTargets = np.tile(np.array([0,1,0,0,0]),(tempminLen,1))
			#tempmTargets = np.tile(np.array([0,0,1,0,0]),(tempminLen,1))
			#temprTargets = np.tile(np.array([0,0,0,1,0]),(tempminLen,1))
			#temppTargets = np.tile(np.array([0,0,0,0,1]),(tempminLen,1))
			
			#tTargets = np.vstack((tTargets, temptTargets));
			#iTargets = np.vstack((iTargets, tempiTargets));
			#mTargets = np.vstack((mTargets, tempmTargets));
			#rTargets = np.vstack((rTargets, temprTargets));
			#pTargets = np.vstack((pTargets, temppTargets));
			
			## We have to check if the new minValue is less than the previous minValue
			if(tempMinValue < minValue):
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#tDelete = np.delete(tData, 1275, 1);
				#print(tDelete.shape)
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				#In this case, we have to truncate the values already stored to the current minValue
				tData = np.vstack((tData[:, :, 0:tempMinValue], temptData));
				iData = np.vstack((iData[:, :, 0:tempMinValue], tempiData));
				mData = np.vstack((mData[:, :, 0:tempMinValue], tempmData));
				rData = np.vstack((rData[:, :, 0:tempMinValue], temprData));
				pData = np.vstack((pData[:, :, 0:tempMinValue], temppData));
				minValue = tempMinValue;
			elif(tempMinValue == minValue):
				#In this case, we don't need truncation
				tData = np.vstack((tData, temptData));
				iData = np.vstack((iData, tempiData));
				mData = np.vstack((mData, tempmData));
				rData = np.vstack((rData, temprData));
				pData = np.vstack((pData, temppData));
			else:
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				tData = np.vstack((tData, temptData[:, :, 0:minValue]));
				iData = np.vstack((iData, tempiData[:, :, 0:minValue]));
				mData = np.vstack((mData, tempmData[:, :, 0:minValue]));
				rData = np.vstack((rData, temprData[:, :, 0:minValue]));
				pData = np.vstack((pData, temppData[:, :, 0:minValue]));
                
    

	##Combine the data by stacking the individual markers and finger data
	lentData = len(tData);
	leniData = len(iData);
	lenmData = len(mData);
	lenrData = len(rData);
	lenpData = len(pData);

	tTargets = np.tile(np.array([1,0,0,0,0]),(lentData,1))
	iTargets = np.tile(np.array([0,1,0,0,0]),(leniData,1))
	mTargets = np.tile(np.array([0,0,1,0,0]),(lenmData,1))
	rTargets = np.tile(np.array([0,0,0,1,0]),(lenrData,1))
	pTargets = np.tile(np.array([0,0,0,0,1]),(lenpData,1))

	Data = np.vstack((tData, iData, mData, rData, pData));
	Targets = np.vstack((tTargets, iTargets, mTargets, rTargets, pTargets));

	filters = CSP(tData, iData, mData, rData, pData);
	filtersArray = np.asarray(filters)
	tData = np.matmul(np.transpose(filtersArray[0]), tData);
	iData = np.matmul(np.transpose(filtersArray[1]), iData);
	mData = np.matmul(np.transpose(filtersArray[2]), mData);
	rData = np.matmul(np.transpose(filtersArray[3]), rData);
	pData = np.matmul(np.transpose(filtersArray[4]), pData);

	DataCSP = np.vstack((tData, iData, mData, rData, pData));
	TargetsCSP = np.vstack((tTargets, iTargets, mTargets, rTargets, pTargets));

	if(shuffleData == True):
		Data, Targets = shuffle(Data, Targets, random_state=0)
		DataCSP, TargetsCSP = shuffle(DataCSP, TargetsCSP, random_state=0)
	return (Data, Targets, DataCSP, TargetsCSP)

def GetCombinedData_FreeForm(relativeFilePath, shuffleData):
	fileNames = ['FREEFORMSubjectB1511112StLRHand.mat' , 'FREEFORMSubjectC1512102StLRHand.mat', 'FREEFORMSubjectC1512082StLRHand.mat'];
	numDatasets = len(fileNames);
	for dataset in range(0, numDatasets):
		print("Processing dataset {} of {}".format(dataset+1, numDatasets))

		fileName = fileNames[dataset];

		file = sio.loadmat(relativeFilePath+'/{}'.format(fileName)) #replace with .mat file name
		header=file['__header__']
		version=file['__version__']
		glob=file['__globals__']
		#ans=file['ans']


		#x=file['x']
		o=file['o'][0][0]
		data=o['data']
		data = np.transpose(data)
		data = data[0:21,:];
		#print(data)
		nS=o['nS'][0][0]
		#values of structure seem to be 2D numpy arrays, if originally a scalar in Matlab.
		#use [0][0] to get scalar.
		#print("Number of samples: {numSamples}".format(numSamples=nS))
		test=o['id'][0] #id value became a 1D array of size 1 for some reason. use [0] to get value
		#print("Dataset ID: {id}".format(id=test))
		chnames=o['chnames'][:,0] #[:,0] converts from 2D array back to 1D array
		#print("Channel names: {channelNames}".format(channelNames=chnames))
		markers = o['marker']
		## The markers are all still individual arrays of size 1x1, so we convert them to an array with single values
		markersArray = []
		for marker in markers:
			markersArray.append(marker[0])
		markersArray = np.asarray(markersArray)

		## Find the starting indeces where the marker changes
		changeIdxs = np.where(np.transpose(markersArray)[:-1] != np.transpose(markersArray)[1:])[0]
		#print("Number of index changes: {idxChanges}".format(idxChanges=changeIdxs.shape[0]))
		## Split the data so that it has its matching marker
		dataSplit = np.array_split(data, changeIdxs[:-1], axis=1)
		splitCount = 0
		for splitData in dataSplit:
			splitCount += 1
		#print("Number of arrays in data split: {num}".format(num=splitCount))
		## Retrieve the marker values for each of the change indeces (changeIdxs)
		markerTargets = markersArray[changeIdxs];
		#print("Number of marker targets: {numTargets}".format(numTargets=markerTargets.shape[0]))

		## To Apply CSP, we first only get the indeces for MI tasks 1 and 2 (left and right hand, respectively.)
		lhIdx = np.where(markerTargets == 1)[0]
		rhIdx = np.where(markerTargets == 2)[0]

		lhIdxMin = GetMinSteps(lhIdx, dataSplit)
		rhIdxMin = GetMinSteps(rhIdx, dataSplit)
		minValues = [lhIdxMin, rhIdxMin]
		#minValues

		#Truncate the data to the min size
		#print(minValue)

		if(dataset == 0):
			
			minValue = np.min(minValues)
			
			lhData = GetData(lhIdx, dataSplit, minValue)
			rhData = GetData(rhIdx, dataSplit, minValue)
			minLen = np.min([len(lhData), len(rhData)])
			
			lhData = lhData[0:minLen]
			rhData = rhData[0:minLen]
			
			
			
		else:
			tempMinValue = np.min(minValues)
			templhData = GetData(lhIdx, dataSplit, tempMinValue)
			temprhData = GetData(rhIdx, dataSplit, tempMinValue)

			tempminLen = np.min([len(templhData), len(temprhData)])
			templhData = templhData[0:minLen]
			temprhData = temprhData[0:minLen]
			
			## We have to check if the new minValue is less than the previous minValue
			if(tempMinValue < minValue):
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				#In this case, we have to truncate the values already stored to the current minValue
				lhData = np.vstack((lhData[:, :, 0:tempMinValue], templhData));
				rhData = np.vstack((rhData[:, :, 0:tempMinValue], temprhData));
				minValue = tempMinValue;
			elif(tempMinValue == minValue):
				#In this case, we don't need truncation
				lhData = np.vstack((lhData, templhData));
				rhData = np.vstack((rhData, temprhData));

			else:
				#print("Previous minValue: {}\tCurrent minValue: {}".format(minValue, tempMinValue));
				#pauseStr = input("Pausing after iteration: " + str(dataset))
				lhData = np.vstack((lhData, templhData[:, :, 0:minValue]));
				rhData = np.vstack((rhData, temprhData[:, :, 0:minValue]));      

	##Combine the data by stacking the individual markers and finger data
	lenlhData = len(lhData);
	lenrhData = len(rhData);

	lhTargets = np.tile(np.array([1,0]),(lenlhData,1))
	rhTargets = np.tile(np.array([0,1]),(lenrhData,1))

	Data = np.vstack((lhData, rhData));
	Targets = np.vstack((lhTargets, rhTargets));

	filters = CSP(lhData, rhData);
	filtersArray = np.asarray(filters)

	lhData = np.matmul(np.transpose(filtersArray[0]), lhData);
	rhData = np.matmul(np.transpose(filtersArray[1]), rhData);

	DataCSP = np.vstack((lhData, rhData));
	TargetsCSP = np.vstack((lhTargets, rhTargets));

	if(shuffleData == True):
		Data, Targets = shuffle(Data, Targets, random_state=0)
		DataCSP, TargetsCSP = shuffle(DataCSP, TargetsCSP, random_state=0)
	return (Data, Targets, DataCSP, TargetsCSP)