import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout

#Prepare the data

data = np.genfromtxt('Schreiber08_IK_grf.csv', delimiter = ',') # numpy.ndarray, (3814, 45)

#Divide the angle and grf data into 4 subsets according to the trial number
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
        angles1.append(aux_ang)
        grf1.append(aux_grf)
    elif row[41] == 2:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        angles2.append(aux_ang)
        grf2.append(aux_grf)
    elif row[41] == 3:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        angles3.append(aux_ang)
        grf3.append(aux_grf)
    elif row[41] == 4:
        aux_ang = []
        aux_grf = []
        for i in ANGLE_INDEX:
            aux_ang.append(row[i])
        for j in GRF_INDEX:
            aux_grf.append(row[j])
        angles4.append(aux_ang)
        grf4.append(aux_grf)

#convert lists to numpy arrays and reshape for model inputs
angles1, angles2, angles3, angles4 = np.array(angles1), np.array(angles2), np.array(angles3), np.array(angles4)
grf1, grf2, grf3, grf4 = np.array(grf1), np.array(grf2), np.array(grf3), np.array(grf4)

angles1 = np.reshape(angles1, (1, angles1.shape[0], angles1.shape[1]))
angles2 = np.reshape(angles2, (1, angles2.shape[0], angles2.shape[1]))
angles3 = np.reshape(angles3, (1, angles3.shape[0], angles3.shape[1]))
angles4 = np.reshape(angles4, (1, angles4.shape[0], angles4.shape[1]))
grf1 = np.reshape(grf1, (1, grf1.shape[0], grf1.shape[1]))
grf2 = np.reshape(grf2, (1, grf2.shape[0], grf2.shape[1]))
grf3 = np.reshape(grf3, (1, grf3.shape[0], grf3.shape[1]))
grf4 = np.reshape(grf4, (1, grf4.shape[0], grf4.shape[1]))

print("A1: ", angles1.shape, ", A2: ", angles2.shape, ", A3: ", angles3.shape, ", A4: ", angles4.shape)
print("G1: ", grf1.shape, ", G2: ", grf2.shape, ", G3: ", grf3.shape, ", G4: ", grf4.shape)

#group 3 largest subsets for training
train_angles = (angles2, angles3, angles4)
train_grf = (grf2, grf3, grf4)


def R_squared(measured, predicted):
    correlation_matrix = np.corrcoef(measured, predicted)
    correlation_xy = correlation_matrix[0,1]
    r_sq = correlation_xy**2
    return r_sq


if True:
    EPOCHS = 4
    BATCH_SIZE = 10
    LEARNING_RATE = 0.01

    GNorm = keras.initializers.GlorotNormal()

    #construct the model
    model = Sequential()
    model.add(SimpleRNN(5, input_dim=5, return_sequences=True, kernel_initializer=GNorm,
                                                            recurrent_initializer=GNorm))
    model.add(Dropout(0.3))
    model.add(Dense(3, kernel_initializer=GNorm))

    rmsprop = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    #compile the model
    model.compile(optimizer=rmsprop, loss= "mse")
    model.summary()

    #train the model
    print("fitting the model...")
    loss = []
    val_loss = []
    for i in range(3):
        history = model.fit(train_angles[i], train_grf[i], epochs= EPOCHS, batch_size= BATCH_SIZE, validation_data=(angles1, grf1))
        loss += history.history['loss']
        val_loss += history.history['val_loss']

    #create diagnostic plot
    plt.plot(loss)
    plt.plot(val_loss)
    title = 'RMSprop: Epoch=' + str(EPOCHS) + ', learning rate=' + str(LEARNING_RATE) + ', batch size=' + str(BATCH_SIZE)
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.show()

    #calculate R squared
    predictions1 = model.predict(angles1)
    predictions2 = model.predict(angles2)
    predictions3 = model.predict(angles3)
    predictions4 = model.predict(angles4)
    Rx1, Ry1, Rz1 = R_squared(grf1[0][...,0], predictions1[0][...,0]), R_squared(grf1[0][...,1], predictions1[0][...,1]), R_squared(grf1[0][...,2], predictions1[0][...,2])
    Rx2, Ry2, Rz2 = R_squared(grf2[0][...,0], predictions2[0][...,0]), R_squared(grf2[0][...,1], predictions2[0][...,1]), R_squared(grf2[0][...,2], predictions2[0][...,2])
    Rx3, Ry3, Rz3 = R_squared(grf3[0][...,0], predictions3[0][...,0]), R_squared(grf3[0][...,1], predictions3[0][...,1]), R_squared(grf3[0][...,2], predictions3[0][...,2])
    Rx4, Ry4, Rz4 = R_squared(grf4[0][...,0], predictions4[0][...,0]), R_squared(grf4[0][...,1], predictions4[0][...,1]), R_squared(grf4[0][...,2], predictions4[0][...,2])
    print("R_squared for validation subset: [{0:.4f}, {1:.4f}, {2:.4f}]".format(Rx1, Ry1, Rz1))
    print("R_squared for training subset 1: [{0:.4f}, {1:.4f}, {2:.4f}]".format(Rx2, Ry2, Rz2))
    print("R_squared for training subset 2: [{0:.4f}, {1:.4f}, {2:.4f}]".format(Rx3, Ry3, Rz3))
    print("R_squared for training subset 3: [{0:.4f}, {1:.4f}, {2:.4f}]".format(Rx4, Ry4, Rz4))
    print("Mean of R_squared: [{0:.4f}, {1:.4f}, {2:.4f}]".format(np.mean([Rx1, Rx2, Rx3, Rx4]), np.mean([Ry1, Ry2, Ry3, Ry4]), np.mean([Rz1, Rz2, Rz3, Rz4])))
    print("SD of R_squared: [{0:.4f}, {1:.4f}, {2:.4f}]".format(np.std([Rx1, Rx2, Rx3, Rx4]), np.std([Ry1, Ry2, Ry3, Ry4]), np.std([Rz1, Rz2, Rz3, Rz4])))