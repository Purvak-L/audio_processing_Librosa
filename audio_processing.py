import IPython.display as ipd
import librosa
import librosa.display
import os
import pandas as pd
import librosa
import glob 

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 


ipd.Audio('./Train/2.wav')

data, sampling_rate = librosa.load('./Train/2.wav')

get_ipython().magic('pylab inline')



plt.figure(figsize=(12, 4))

librosa.display.waveplot(data,sr=sampling_rate)

train = pd.read_csv('train.csv')


i = random.choice(train.index)
audio_name = train.ID[i]
class_name = train.Class[i]
data_dir = './'
path = os.path.join(data_dir,'Train/',str(audio_name)+'.wav')
print(path)
print('Class: ', class_name)
x, sr = librosa.load(path)

plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr)

print(train.Class.value_counts())


test = pd.read_csv('./Test.csv')
test['Class'] = 'jackhammer'
test.to_csv('sub01.csv', index=False)


def parser(row):
    
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    #Mel Frequency Cepstral Coefficient (MFCC) 
    #The first step in any automatic speech recognition system is to extract features i.e. 
    #identify the components of the audio signal that are good for identifying #
    #the linguistic content and discarding all 
    #the other stuff which carries information like background noise, emotion etc. 
    
    mfcc = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    
    feature = mfcc
    label = row.Class
    
    return [feature,label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']


from sklearn.preprocessing import LabelEncoder
#import np_utils
import numpy as np
from keras.utils import np_utils




X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))

num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

tra = X[0:5000]
valx = X[5000:7000]

tr = y[0:5000]
valy = y[5000:7000]


# In[ ]:


model.fit(tra,tr, batch_size=32, epochs=100, validation_data=(valx,valy))




''''
#Validation data

Test = pd.read_csv('Test.csv')
val_path = './Test'
def parser(row):
    
    X, sample_rate = librosa.load(val_path,res_type='kaiser_fast')
    #Mel Frequency Cepstral Coefficient (MFCC) 
    #The first step in any automatic speech recognition system is to extract features i.e. 
    #identify the components of the audio signal that are good for identifying #
    #the linguistic content and discarding all 
    #the other stuff which carries information like background noise, emotion etc. 
    
    mfcc_val = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    
    feature = mfcc_val
    label = row.Class
    
    return [feature,label]

temp_val = Test.apply(parser, axis=1)
temp_val.columns = ['feature', 'label']

valx = np.array(temp_val.feature.tolist())
valy = np.array(temp_val.label.tolist())

'''


