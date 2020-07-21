import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

#Prepare the data

data = np.genfromtxt('Schreiber08_IK_grf.csv', delimiter = ',') # numpy.ndarray, (3814, 45)

#Divide the data into 4 batches according to the trial no.
#The batches are initialized as empty lists, but replaced with np.ndarrays later
angles1 = []
angles2 = []
angles3 = []
angles4 = []
grf1 = []
grf2 = []
grf3 = []
grf4 = []

#no. of the trial at index 41
ANGLE_INDEX = (13, 14, 15, 16, 17)
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

#stack three angle sets for training
angles = np.vstack((angles1, angles2))
angles = np.vstack((angles, angles3))
print("angle shape: ", angles.shape)
train = np.reshape(angles, (1,  angles.shape[0], angles.shape[1]))
print("reshaped: ", train.shape)

#stack three grf sets for training
grf = np.vstack((grf1, grf2))
grf = np.vstack((grf, grf3))
print("grf shape: ", grf.shape)
'''goal = np.reshape(grf, (grf.shape[0], grf.shape[1], 1))'''
goal = np.reshape(grf, (1, grf.shape[0], grf.shape[1]))
print("reshaped: ", goal.shape)

#one set of angles and grf for testing
test_angles = np.reshape(angles4, (1, angles4.shape[0], angles4.shape[1]))
test_grf = np.reshape(grf4, (1, grf4.shape[0], grf4.shape[1]))


if True:
#construct the model
    EPOCHS = 100
    BATCH_SIZE = 10

    model = Sequential(SimpleRNN(3, input_shape=(None,5), return_sequences=True))
    model.add(Dense(3))

    #different optmimizers
    adam = keras.optimizers.Adam(learning_rate=0.5)
    rmsprop = keras.optimizers.RMSprop(learning_rate=0.5)
    adadelta = keras.optimizers.Adadelta(learning_rate=0.5)

    model.compile(optimizer=rmsprop, loss="mse")
    model.summary()
    print("fitting the model...")
    model.fit(train, goal, epochs= EPOCHS, batch_size= BATCH_SIZE)

    print("evaluating...")
    results = model.evaluate(test_angles, test_grf, batch_size= BATCH_SIZE)

