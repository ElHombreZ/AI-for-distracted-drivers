#Copying CNN model 

X_train, y_train_str = get_train_data(flatten=True)
X_test, y_test_str = get_test_data(flatten=True)

X_train = X_train.reshape([-1, 64, 64, 3])
X_test = X_test.reshape([-1, 64, 64, 3])

# save string versions of labels
y_train = y_train_str
y_test = y_test_str

# convert labels into numpy vectors (one-hot encoding!)
y_train = label_to_numpy(y_train)
y_test = label_to_numpy(y_test)

dense = DenseClassifier(hidden_layer_sizes = (128,64))
cnn = CNNClassifier(num_hidden_layers = 5)

dense.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])
cnn.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])

print('Dense')
plot_acc(dense.history)

print('CNN')
plot_acc(cnn.history)
