# Training model (as before)
X_train, y_train_str = get_train_data(flatten=True)
X_test, y_test_str = get_test_data(flatten=True)

X_train = X_train.reshape([-1, 64, 64, 3])
X_test = X_test.reshape([-1, 64, 64, 3])

y_train = y_train_str
y_test = y_test_str

y_train = label_to_numpy(y_train)
y_test = label_to_numpy(y_test)

from tensorflow.keras.applications.vgg16 import VGG16

vgg_expert = VGG16(weights = 'imagenet', include_top = False, input_shape = (64, 64, 3))

vgg_model = Sequential()
vgg_model.add(vgg_expert)

vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(512, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(4, activation = 'softmax'))

vgg_model.compile(loss = 'categorical_crossentropy',
          optimizer = optimizers.SGD(learning_rate=1e-4, momentum=0.95),
          metrics=['accuracy'])

vgg_model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), shuffle = True, callbacks = [monitor])



# Evaluation and matrix
from sklearn.metrics import accuracy_score, confusion_matrix
predictions = vgg_model.predict(X_test)
predictions = np.argmax(predictions,axis=1)

final_labels = []
for label in y_test_str:
  if label == 'Attentive':
    final_labels.append(0)
  else:
    final_labels.append(1)

binary_predictions = []
for label in predictions:
  if label == 0:
    binary_predictions.append(0)
  else:
    binary_predictions.append(1)

# Checking if the lists have been defined allows the whole notebook to execute
# even if this cell has not been completed
if binary_predictions and final_labels:
  print('Accuracy is %d %%'%(accuracy_score(final_labels, binary_predictions)*100.0))
  
#consfusion matrix
confusion = confusion_matrix(final_labels, binary_predictions)
print(confusion)

tp  = confusion[1][1]
tn  = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]

print('True positive: %d'%tp)
print('True negative: %d'%tn)
print('False positive: %d'%fp)
print('False negative: %d'%fn)

mport seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion, annot = True, fmt = 'd', cbar_kws={'label':'count'});
plt.ylabel('Actual');
plt.xlabel('Predicted');

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print('Precision:', precision)
print('Recall:', recall)
