from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation
import numpy as np
import random

def get_val(x):
	x=[float(ord(c.lower())) for c in x]
	x_sum=0.
	for c in x: x_sum+=c
	return float(x_sum)

def func_moa(x):
	return (int(2.**float((x*4310.)%100))%101)

def get_char_indices(filename):
	with open(filename, 'r') as r:
		data=r.readlines()

	txt=""
	for n1 in data:
		for n2 in data:
			if n1!=n2: 
				n1_t=n1.replace('\n', '').lower()
				n2_t=n2.replace('\n', '').lower()
				txt+=n1_t+n2_t
		data.remove(n1)

	chars=list(set(txt))
	char_indices = dict((c, i) for i, c in enumerate(chars))

	return char_indices

def recompile_model(weights_filename):
	maxlen=128
	model=Sequential()
	model.add(Dense(8, kernel_initializer="normal", input_dim=maxlen))
	model.add(Dense(1, kernel_initializer="normal"))

	model.load_weights(weights_filename)
	model.compile(loss='mean_squared_error', optimizer="adam")
	return model

def use_model(model, filename, n1, n2):
	maxlen=128
	char_indices=get_char_indices(filename)

	if get_val(n1)>get_val(n2): pair=n1.lower()+n2.lower()
	else: pair=n2.lower()+n1.lower()

	test_array=np.ones(maxlen, dtype=np.int64) * -1
	for t, char in enumerate(pair[-maxlen:]):
		test_array[(maxlen - 1 - t)]=char_indices[char]
	pred=(model.predict(np.array(test_array).reshape(1,maxlen))[0][0])
	return func_moa(pred)



