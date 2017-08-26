# Audio processing - Librosa
Basic audio processing using python library librosa. Extracted features using MFCC.
The output of mfcc was later plugged into deep learning model constructed using keras.
The input was classified into 10 classes.

Class               values

jackhammer          668
engine_idling       624
siren               607
drilling            600
air_conditioner     600
dog_bark            600
street_music        600
children_playing    600
car_horn            306
gun_shot            230

Dependencies :

Librosa (pip install librosa): If librosa.display isn't working, import librosa.display instead of just importing librosa
keras (pip install keras)

Dataset - 'https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU?usp=sharing'

The code was inspired by analytics vidya's tutorial on audio processing (https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)


To read more about MFCC - http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
