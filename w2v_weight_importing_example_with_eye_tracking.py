from __future__ import print_function
from keras import backend as K
import numpy as np
import gensim
np.random.seed(1338)  # for reproducibility

from keras.preprocessing import sequence
from keras.regularizers import l1,l2, activity_l1,activity_l2
from keras.utils import np_utils
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adagrad
from keras.layers.normalization import BatchNormalization
from pre_Process_Helper import preProcess


#from pre_Process_Helper import preProcess

#X, y, s = preProcess()#np.load("tensor.npy", encoding = "latin1")
X,y,s = preProcess(True,"bl i", "1")
rp = np.random.permutation(X.shape[0])
X=X[rp]
y=y[rp]
s=s[rp]

firstR=np.zeros(X.shape[0])

for x in range(X.shape[0]):
 firstR[x] = np.where(s[x,:]==1)[0][0]
 print(firstR[x])

resplen = [np.where(s[x]==1)[0][-1] for x in range(len(s))]
q75, q25 = np.percentile(resplen, [75 ,25])
iqr = q75 - q25
resplenf = [np.where(s[x]==1)[0][0] for x in range(len(s))]
q75f, q25f = np.percentile(resplenf, [75 ,25])
iqrf = q75f - q25f
keep = np.logical_and(resplen < q75 + 1.5*iqr, resplen > q25f - 1.5*iqrf)

X=X[keep]
y=y[keep]
s=s[keep]

sampfreq = 60
sample = []
for stu in range(len(X)):
      print(stu)
      print(sum(s[stu]))
      a=np.zeros(X.shape[1],dtype='bool')
      a[range(0,X.shape[1],sampfreq)]=True
      a[np.where(X[stu]==0)]=False
      a[np.where(s[stu]==1)[0]]=True
#     a[np.where(s[stu]==1)[0][0]+1:]=False
      sample.append(a)

sampmax=[np.sum(i) for i in sample]
print(sampmax)
for stu in range(len(X)):
      X[stu]=np.pad(X[stu][sample[stu]],(0,X.shape[1]-sampmax[stu]),
                    'constant',constant_values=0)
      y[stu,:,0]=np.pad(y[stu,:,0][sample[stu]],(0,X.shape[1]-sampmax[stu]),
                    'constant',constant_values=0)
      s[stu]=np.pad(s[stu][sample[stu]],(0,X.shape[1]-sampmax[stu]),
                    'constant',constant_values=0)
X=X[:,0:np.max(sampmax)]
y=y[:,0:np.max(sampmax)]
s=s[:,0:np.max(sampmax)]

tr = range(round(len(X)*.71))
te = range(len(tr),len(X))

print('average y[te].pctcor: ',np.mean(y[te][s[te]]),'num ans: ',np.sum(y[s]), 'rmse baseline: ',np.sqrt(np.mean((np.mean(y[tr][s[tr]])-y[te][s[te]])**2)))
print('X shape: ',X.shape)

# begin w2v procedure

print(X.shape)
sen=[]
for stu in range(len(X[tr])):
      try:
            sen.append([str(i) for i in X[tr][stu][0:np.where(X[tr][stu] != 0)[0][-1]]])
      except:
            pass
print(len(sen))

esize = 12
model=gensim.models.Word2Vec(sen,size=esize,window=round(120/sampfreq),
                             sg=1,min_count=1,workers=20,iter=50)


# print(X[[tr]].shape)
# print(y[[tr]].shape)
# print(s[[tr]].shape)

lsize = 64

vocab = np.max(X)
print('vocab size',vocab)
embedding_weights=np.random.rand(vocab+1,esize)
for i in np.unique(X[tr]):
      try:
            embedding_weights[i]=model[str(i)]
      except:
            pass
model = Sequential()
model.add(Embedding(vocab+1,esize,mask_zero=1,
                    input_length=X.shape[1], weights=[embedding_weights]))
model.add(LSTM(lsize, return_sequences=True, activation='tanh'))
#model.add(Dropout(0.20))
model.add(LSTM(lsize, return_sequences=True, activation='tanh'))
#model.add(Dropout(0.20))
model.add(LSTM(lsize, return_sequences=True, activation='tanh'))
#model.add(Dropout(0.20))
model.add(LSTM(lsize, return_sequences=True, activation='tanh'))
model.add(TimeDistributed(Dense(1, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01))))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
	      optimizer='rmsprop',
#              optimizer=Adagrad(lr=0.001, clipnorm=0.1),
	      sample_weight_mode='temporal')
#	      metrics=['recall_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
# checkpointer = ModelCheckpoint(filepath='rnn_model_all.hdf5',
#                                verbose=0,
#                                save_best_only=True)

# model.load_weights('rnn_model_all.hdf5')
print('adamax')
for x in range(10000):
      print(x)
      model.fit(X[tr],y[tr], batch_size=2,nb_epoch=1, sample_weight=s[tr])
      p = model.predict(X[te])
      tot = p.shape[0]*p.shape[1]
      toty = y[te].shape[0]*y[te].shape[1]
      tots = s[te].shape[0]*s[te].shape[1]
      p = p.reshape([tot,1])
      y2 = y[te].reshape([toty,1])
      s2 = s[te].reshape([tots,1])
      acc = np.mean([np.round(p[[s2]])==y2[[s2]]])
      print('Accuracy: ',acc, 'RMSE: ',np.sqrt(np.mean((p[s2]-y2[s2])**2)))
      print('prediction min',np.min(p[s2]),'max',np.max(p[s2]))
#      print(np.sqrt(np.mean((p[s2]-y2[s2])**2)))
#      print(np.mean((p[s2]-y2[s2])**2))

#get_3rd_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[0].output])

#layer_output = get_3rd_layer_output([X[tr,0:X.shape[1]-1]])[0]
#losize = len(layer_output)

#print(layer_output.shape)
#np.savetxt('outputs.txt',layer_output.reshape([losize*layer_output.shape[1],lsize])
#           [sw[tr,0:X.shape[1]-1].reshape([losize*layer_output.shape[1]])]
#           ,delimiter='\t')
