import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

#Prepare the data

data = np.genfromtxt('Schreiber08_IK_grf.csv', delimiter = ',') # numpy.ndarray, (3814, 45)

#Divide the data into 4 batches according to the trial no.
angles1 = []
angles2 = []
angles3 = []
angles4 = []
grf1 = []
grf2 = []
grf3 = []
grf4 = []

for row in data:
    if row[41] == 1:
        angles1.append((row[14], row[15], row[16], row[17], row[18]))
        grf1.append((row[42], row[43], row[44]))
    elif row[41] == 2:
        angles2.append((row[14], row[15], row[16], row[17], row[18]))
        grf2.append((row[42], row[43], row[44]))
    elif row[41] == 3:
        angles3.append((row[14], row[15], row[16], row[17], row[18]))
        grf3.append((row[42], row[43], row[44]))
    elif row[41] == 4:
        angles4.append((row[14], row[15], row[16], row[17], row[18]))
        grf4.append((row[42], row[43], row[44]))

for i in angles1:
    print(i[0], " // ", i[1], " // ", i[2], " // ", i[3], " // ", i[4])

c=0
if c == True:
#Construct the model

    model = Sequential(SimpleRNN(3, input_shape=(5,1)))
    model.summary()