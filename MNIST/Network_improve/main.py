from keras.optimizers import SGD,Adagrad,Adam,Adamax,Adadelta,RMSprop,Nadam
from keras.callbacks import EarlyStopping

from data import load_data
from model import get_model

batch_size = 128
nb_epoch = 25

# Load data
(X_train, y_train, X_test, y_test) = load_data()

# Load and compile model
model = get_model()

model.compile(loss='categorical_crossentropy', 
			  optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
#optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True)
#optimizer=Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
#optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
#optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))


score = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy:", score[1])
