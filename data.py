import numpy as np
import theano
import os.path
import subprocess

import utils

TEST_PATH = 'cb513+profile_split1.npy.gz'
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'

def datagenerator(proteins,batch_size):
	alldata,labels = get_train()
	x_data = np.zeros((batch_size,15,22))
	y_data = np.zeros(batch_size)
	while True:
		idx = 0
		for protein in range(proteins):
			for amino in range(700):
				if labels[protein,amino]!=8:
					y_data[idx]=labels[protein,amino]
					for window in range(15):
						if amino+window-7>=700:
							for features in range(21):
								x_data[idx,window,features] = 0
							x_data[idx,window,21]=1
						else:
							for features in range(22):
								x_data[idx,window,features] = alldata[protein,amino+window-7,features]
					idx+=1
					if idx==batch_size:
						idx=0
						x_data_ret = x_data.reshape(batch_size,15*22)
						yield x_data_ret.reshape(batch_size,22*15,1),y_data.reshape(batch_size)
				else:
					continue

def dataconversion(proteins,matrixdata,labels):
	x_data = np.zeros((32,15,22))
	y_data = np.zeros(32)
	x_data_list = []
	y_data_list = []
	idx=0
	for protein in range(proteins):
		for amino in range(700):
			if labels[protein,amino]!=8:
				y_data[idx]=labels[protein,amino]
				for window in range(15):
					if amino+window-7>=700:
						for features in range(21):
							x_data[idx,window,features] = 0
						x_data[idx,window,21]=1
					else:
						for features in range(22):
							x_data[idx,window,features] = matrixdata[protein,amino+window-7,features]
				idx+=1
				if idx==32:
					idx=0
					x_data_tmp = x_data.reshape(32,15*22)
					x_data_list.append(x_data_tmp.copy())
					y_data_list.append(y_data.copy())
			else:
				continue
	x_data_ret= np.asarray(x_data_list)
	y_data_ret= np.asarray(y_data_list)
	return x_data_ret.reshape(x_data_ret.shape[0]*x_data_ret.shape[1],15*22),y_data_ret.reshape(y_data_ret.shape[0]*y_data_ret.shape[1])





def get_train():
	print("Loading train data ...")
	X_in = np.load(TRAIN_PATH,mmap_mode='r')
	print(X_in.shape)
	X = np.reshape(X_in,(5534,700,57))
	del X_in
	labels = X[:,:,22:31]

	a = np.arange(0,22)
	#b = np.arange(35,57)
	c = np.hstack((a))
	X = X[:,:,c]

	proteins = np.size(X,0)
	aminolength = np.size(X,1)
	learningdata = np.size(X,2)
	num_classes = 9

	X = X.astype(float)
	vals = np.arange(0,9)
	labels_new = np.zeros((proteins,aminolength))
	for i in xrange(np.size(labels,axis=0)):
		labels_new[i,:] = np.dot(labels[i,:,:], vals)
	labels_new = labels_new.astype(int)
	labels = labels_new
	return X, labels

def get_test():
	print("Loading test data ...")
	X_in = np.load(TEST_PATH)
	X = np.reshape(X_in,(514,700,57))
	del X_in
	labels = X[:,:,22:31]

	a = np.arange(0,22)
	#b = np.arange(35,57)
	c = np.hstack((a))
	X = X[:,:,c]

	proteins = np.size(X,0)
	aminolength = np.size(X,1)
	learningdata = np.size(X,2)
	num_classes = 9

	X = X.astype(float)

	vals = np.arange(0,9)
	labels_new = np.zeros((proteins,aminolength))
	for i in xrange(np.size(labels,axis=0)):
		labels_new[i,:] = np.dot(labels[i,:,:], vals)
	labels_new = labels_new.astype(int)
	labels = labels_new
	return X, labels
def crossvalidation(parttoreturn,k):
	print("Loading test data ...")
	X_in = np.load(TEST_PATH)
	X = np.reshape(X_in,(514,700,57))
	del X_in
	labels = X[:,:,22:31]

	a = np.arange(0,22)
	#b = np.arange(35,57)
	c = np.hstack((a))
	X = X[:,:,c]

	proteins = np.size(X,0)
	aminolength = np.size(X,1)
	learningdata = np.size(X,2)
	num_classes = 9
	X = X.astype(float)
	vals = np.arange(0,9)
	labels_new = np.zeros((proteins,aminolength))
	for i in xrange(np.size(labels,axis=0)):
		labels_new[i,:] = np.dot(labels[i,:,:], vals)
	labels_new = labels_new.astype(int)
	labels = labels_new

	X = np.array_split(X,k)
	labels = np.array_split(labels,k)
	xreturn = []
	yreturn = []
	for i in range(k):
		if i != parttoreturn:
			xreturn.append(X[i].copy())
			yreturn.append(labels[i].copy())
	xreturn = np.concatenate(xreturn)
	yreturn = np.concatenate(yreturn)

	return xreturn,yreturn,X[parttoreturn],labels[parttoreturn]