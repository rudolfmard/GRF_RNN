import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

#Prepare the data

data = np.genfromtxt('Schreiber08_IK_grf.csv', delimiter = ',') # numpy.ndarray, (3814, 45)

#Divide the data into 4 batches according to the trial no.
#The batches (1 array for angles and 1 for GRF) are initialized as empty lists, but replaced with np.ndarrays later
angles1 = []
angles2 = []
angles3 = []
angles4 = []
grf1 = []
grf2 = []
grf3 = []
grf4 = []

#no. of the trial at index 41
ANGLE_INDEX = (14, 15, 16, 17, 18)
GRF_INDEX = (42, 43, 44)
for row in data:
    if row[41] == 1:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        if len(angles1) == 0:
            angles1 = np.array([aux_ang])
            grf1 = np.array([aux_grf])
        else:
            angles1 = np.vstack((angles1, np.array(aux_ang)))
            grf1 = np.vstack((grf1, np.array(aux_grf)))
    elif row[41] == 2:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        if len(angles2) == 0:
            angles2 = np.array([aux_ang])
            grf2 = np.array([aux_grf])
        else:
            angles2 = np.vstack((angles2, np.array(aux_ang)))
            grf2 = np.vstack((grf2, np.array(aux_grf)))
    elif row[41] == 3:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        if len(angles3) == 0:
            angles3 = np.array([aux_ang])
            grf3 = np.array([aux_grf])
        else:
            angles3 = np.vstack((angles3, np.array(aux_ang)))
            grf3 = np.vstack((grf3, np.array(aux_grf)))
    elif row[41] == 4:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        if len(angles4) == 0:
            angles4 = np.array([aux_ang])
            grf4 = np.array([aux_grf])
        else:
            angles4 = np.vstack((angles4, np.array(aux_ang)))
            grf4 = np.vstack((grf4, np.array(aux_grf)))

print("A1: ", angles1.shape, ", A2: ", angles2.shape, ", A3: ", angles3.shape, ", A4: ", angles4.shape)
print("G1: ", grf1.shape, ", G2: ", grf2.shape, ", G3: ", grf3.shape, ", G4: ", grf4.shape)

if False:
#construct the model

    model = Sequential(SimpleRNN(3, input_shape=(5,1)))
    model.summary()