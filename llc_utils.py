from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

def get_val_pair(x, y):
	x=[float(ord(c.lower())) for c in x]
	y=[float(ord(c.lower())) for c in y]

	x_sum=y_sum=0.
	for c in x: x_sum+=c
	for c in y: y_sum+=c

	val=float(x_sum+y_sum)
	return val


def generate_data(filename):

	with open(filename, 'r') as r:
		data=r.readlines()

	pairs=[]
	vals=[]
	txt=""
	n_pairs=0
	for n1 in data:
		for n2 in data:
			if n1!=n2: 
				n1_t=n1.replace('\n', '').lower()
				n2_t=n2.replace('\n', '').lower()
				vals.append(get_val_pair(n1_t, n2_t))
				txt+=n1_t+n2_t
				pairs.append(n1_t+n2_t)
				n_pairs=n_pairs+1
		data.remove(n1)

	chars=list(set(txt))
	char_indices = dict((c, i) for i, c in enumerate(chars))

	maxlen=128
	X=np.ones((n_pairs, maxlen), dtype=np.int64) * -1
	y=np.array(vals)/max(vals)

	for i, pair in enumerate(pairs):
		for t, char in enumerate(pair[-maxlen:]):
			X[i, (maxlen - 1 - t)]=char_indices[char]

	return X, y


def create_model():
	maxlen=128
	model=Sequential()
	model.add(Dense(8, kernel_initializer="normal", input_dim=maxlen))
	model.add(Dense(1, kernel_initializer="normal"))

	model.compile(loss='mean_squared_error', optimizer="adam")

	return model

def train_model(model, X, y):

	earlystop_cb = EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto')
	model.fit(X, y, epochs=4, batch_size=32, shuffle=True, callbacks=[earlystop_cb])

	return model

def save_model_weights_to_filename(model, filename):
	model.save(filename)


