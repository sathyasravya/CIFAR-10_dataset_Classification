import os
import sys
import numpy as np
import math
from PIL import Image
from numpy import zeros
from numpy import linalg as LA

def softmaxfn(row):
    l = len(row)
    y = []
    sum=0
    for i in range(l):
        sum +=np.exp(row[i])
    for i in range(l):
        y.append(np.exp(row[i])/sum)
    return y
X = []
count=0

train_file_path = sys.argv[1]
test_file_path =sys.argv[2]
Nf = 32
numcl = 8
data =dict()
#data = {int:[]}
finlabel = dict()
dataclassnum = dict()#{int:int}
lines_train=[]
count =0
for line in open(train_file_path,"r"):
    x_row=[]
    line_content = line.split(" ")
    lines_train.append(line_content[1])
    #use_ind = path_lines_train[line]
    im = Image.open(line_content[0]).convert('L')
    pix = np.asarray(im,dtype = "int32")
    r,c = np.shape(pix)
    #pixf = np.resize(pix,(256,256)).reshape(1,65536)
    for i in range(r):          #reducing 256*256 into 1D
        for j in range(c):
            x_row.append(pix[i][j])
    X.append(x_row)
    #data.key(line)=line_content[1]
    use=line_content[0].split('/')
    use2=use[-1].split('_')
    dataclassnum[count]=int(use2[0]) # class number of row line in X
    finlabel[int(use2[0])]=line_content[1] # label of that class in X
    #print(dataclassnum[count])
    #print('count is ',count)
    #if(count==19):
    #    print('hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
    #print(len(X[0]))
    #print(len(X))
    ## n by numcl matrix
    count+=1

X = np.matrix(X)
n,d = X.shape
key_matrix=np.zeros([n,numcl],dtype='int32')
for count in range(n):
    key_matrix[count][dataclassnum[count]]=1
#print(X.shape)
#print(dataclassnum)
#print('-----------------------------')
#print(finlabel)
#mean_col_X=[]
#mean_col_X = np.mean(X,axis=0)
feamean = []
feamean = np.mean(X,axis=0)
feamean = np.matrix(feamean)
def princ_comp_anal(X,n,d,Nf,feamean):

    for i in range(n):
        X[i,:] = X[i,:]- feamean #np.mean(X,axis=0)
    XT = np.transpose(X)
    cov_mat = np.matmul(X,XT)
    eigval,eigvec1 = LA.eigh(cov_mat)
    p = np.size(eigvec1, axis =0)
    #sorting
    idx = np.argsort(eigval)
    idx = idx[::-1]
    #<object_name>[<start_index>, <stop_index>, <step>]
    eigvec1 = eigvec1[:,idx]
    eigval = eigval[idx]
#    print(eigval.shape)
#   print('===============eigval matrix=========')
#    print(eigval)
    eigvec = eigvec1[:,:Nf]
    eigvec = np.matmul(XT,eigvec)
    eigvr,eigvc = eigvec.shape
    normeig=np.zeros([eigvr,eigvc])
    normeig = np.transpose(eigvec)
    a, b = np.shape(normeig)
    ##normalising eigen vectors
    for i in range(a):
    	ss = 0
    	for j in range(b):
    		ss += normeig[i,j]**2
    	ss = np.sqrt(ss)
    	for j in range(b):
    		normeig[i,j] /= ss
    normeig = np.transpose(normeig)
    z = np.matmul(X,normeig) # nd dk newx
    proj = np.matmul(z,np.transpose(normeig)) # nk  kd  oldx
    return z,normeig
#for i in open(test_file_path,"r")
newx,neig=princ_comp_anal(X,n,d,Nf,feamean)
line =0
#print(dataclassnum)
for line in dataclassnum:
#    print(dataclassnum[line])
    ap = []
    for it in range(Nf):
        ap.append(newx[line,it])
#    print('yooooooooooooooooooooooooooooooooooooooooooo')
    #print(ap)
    if (dataclassnum[line]) in data:
        data[dataclassnum[line]].append(ap)
    else:
        data[dataclassnum[line]]=[]
        data[dataclassnum[line]].append(ap)
W = np.zeros([Nf,numcl],dtype='int32')#dim * classnum
#intermat =
# n dim  * dim * classnum
intermat  = np.matmul(newx,W)
n,classnum = intermat.shape
print(n,classnum)
soft_ar = []

for i in range(n):
    #for j in range(classnum):
    soft_ar1 = []
    use = np.asarray(softmaxfn(intermat[i,:],dtype = "int32")
    #print(use.shape)
    for j in range(classnum):
        soft_ar1.append(use[0,j])
    #print(soft_ar1)
    soft_ar.append(soft_ar1)
print(len(soft_ar1))
print(len(soft_ar))
#soft_ar =np.matrix(soft_ar)
#print(len(soft_ar))
print(len(soft_ar[0]))
O = np.matrix(soft_ar)
