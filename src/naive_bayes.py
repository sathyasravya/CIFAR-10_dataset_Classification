import os
import sys
import numpy as np
import math
from PIL import Image
from numpy import zeros
from numpy import linalg as LA

X = []
count=0

train_file_path = sys.argv[1]
test_file_path =sys.argv[2]
Nf = 32
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
    count+=1
X = np.matrix(X)
n,d = X.shape
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
#    print(data[dataclassnum[line]])
#print('-----------------===========================================---------------')
#print(data)
nu = {int: []}
cov = {int: []}
prob_class = {int :[]}
clss=0
#print(data)
for clss in data:
    a= len(data[clss])
#    print(a , clss)
    nu[clss]= np.mean(data[clss], axis = 0)
    cov[clss] = np.var(data[clss], axis = 0)
    prob_class[clss]=a/n

n,dim = newx.shape
d,dim = neig.shape
newxt = np.transpose(newx)

X2 = []
ans_cls=[]
numtf=0
correct=0
for l in open(test_file_path,"r"):
    l = l.split('\n')
#    print(l[0])
    ima = Image.open(l[0]).convert('L')
    #ima.show()
    pix = np.asarray(ima,dtype = "int32")
    r,c = np.shape(pix)
    x2_row=[]
    for i in range(r):          #reducing 256*256 into 1D
        for j in range(c):
            x2_row.append(pix[i][j]-feamean[0,j])
    x2_row =np.matrix(x2_row)
    x2_row=np.matmul(x2_row,neig)
    #x2_row=x2_row-mean_col_newx
    #print(x2_row)
    cls2=0
    mle = -10e+24
    for cls2 in data:
        term1=0
        term2=0
 #       print(x2_row.shape)
 #       print(nu[cls2].shape)
        for i in range(Nf):
            term1+=-0.5*(x2_row[0,i]-nu[cls2][i])**2
            term2+=-0.5*(math.log(2*3.14*cov[cls2][i]))
        gi = term1+term2+prob_class[cls2]
#   print(gi/10e+16 , cls2)
        if(gi>=mle):
            mle = gi
            gindx = cls2
    ans_cls.append(gindx)
 #   print(ans_cls[numtf])
    print(finlabel[ans_cls[numtf]].split('\n')[0])
    index=(int(((l[0].split('/'))[-1]).split('_')[0]))
    if(finlabel[index] ==finlabel[ans_cls[numtf]]):
        correct+=1
    numtf+=1
#print(correct/numtf)
   #l+=1
