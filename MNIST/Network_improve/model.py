from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten,Input, Dense

def get_model():
    model = Sequential()
    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10
#CNN ->0.87 with 20 epoches, 102,106 parameters
    model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(input_shape),
                 padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(BatchNormalization())
    #initializers.random_normal(stddev=0.01),kernel_initializer='random_uniform'
    model.add(Dense(128, activation='relu', kernel_initializer=initializers.Orthogonal(gain = 5.0),kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax',kernel_initializer=initializers.Orthogonal(gain = 5.0),kernel_regularizer=regularizers.l2(0.01)))

    print(model.summary())

    return model


"""
def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10

    hidden_size = 256

    inp = Input(shape=input_shape)
    flat = Flatten()(inp)
    b_norm = BatchNormalization()(flat)
    hidden_1 = Dense(hidden_size, activation='sigmoid',kernel_initializer=initializers.Orthogonal(gain = 5.0), kernel_regularizer=regularizers.l2(0.005))(b_norm)
    #hidden_1 = Dense(hidden_size, activation='sigmoid',kernel_initializer=initializers.he_uniform())(flat)
    drop_1 = Dropout(0.25)(hidden_1)
    hidden_2 = Dense(hidden_size, activation='sigmoid',kernel_initializer=initializers.Orthogonal(gain = 5.0),kernel_regularizer=regularizers.l2(0.005))(drop_1)
    #hidden_2 = Dense(hidden_size, activation='sigmoid',kernel_initializer=initializers.he_uniform())(drop_1)
    drop_2 = Dropout(0.25)(hidden_2)
    out = Dense(nb_classes, activation='softmax')(drop_2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model
"""

if __name__ == '__main__':

    model = get_model()


