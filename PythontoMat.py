import scipy.io as sio
file = sio.loadmat('CLA-SubjectJ-170508-3St-LRHand-Inter.mat') #replace with .mat file name
#file is a dictionary
for instance in file:
    print(instance)
header=file['__header__']
version=file['__version__']
glob=file['__globals__']
ans=file['ans']
x=file['x']

o=file['o'][0][0]
#Matlab structures converted into 2D structured numpy array.
#Since o in Matlab is 1x1 structure the numpy array is size (1,1).
#[0][0] is to access the structure so don't have to deal with array

print(o.dtype) #prints all the fields names of the structure
data=o['data']
print(data)
nS=o['nS'][0][0]
#values of structure seem to be 2D numpy arrays, if originally a scalar in Matlab.
#use [0][0] to get scalar.

test=o['id'][0] #id value became a 1D array of size 1 for some reason. use [0] to get value
chnames=o['chnames'][:,0] #[:,0] converts from 2D array back to 1D array
print(chnames[0]) #however, each element is an array still


