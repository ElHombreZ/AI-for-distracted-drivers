metadata = pkg.get_metadata(metadata_path)

#Visualize data
sns.countplot(x = 'class', data = metadata)
metadata.info()
sns.countplot(x = 'split', data = metadata)
sns.countplot(x = 'split', hue = 'class', data = metadata)

#training
X_train, y_train = get_train_data()
image = X_train[0,:] #this gets the first image in our train_data array
image_label = y_train[0] #this gets the first label in our train_labels array

#info on image
image = image.reshape(64, 64, 3)
print('Our image is stored as %s in Python'%type(image))
print('Our image has dimensions of (%d, %d, %d)'%image.shape)
print('Our image has label %s'%image_label)

plot_one_image(image, labels=[image_label])

#customizable for about 7700 images, shows image and label
for i in range(0,7000,1200):
  image = X_train[i]
  image_label = y_train[i]
  plot_one_image(image, labels=[image_label])

#Visual differences between images
radio_X_train = X_train[y_train=='UsingRadio'] #grab all images whose corresponding label is 'UsingRadio'
attentive_X_train = X_train[y_train=='Attentive'] #etc.
coffee_X_train = X_train[y_train=='DrinkingCoffee']
mirror_X_train = X_train[y_train=='UsingMirror']

radio_y_train = y_train[y_train=='UsingRadio'] #grab all images whose corresponding label is 'UsingRadio'
attentive_y_train = y_train[y_train=='Attentive'] #etc.
coffee_y_train = y_train[y_train=='DrinkingCoffee']
mirror_y_train = y_train[y_train=='UsingMirror']

#UsingRadio images
for i in range(500,520):
  image = radio_X_train[i]
  image_label = radio_y_train[i]
  plot_one_image(image, labels=[image_label])

#DrinkingCoffee images
for i in range(500,520):
  image = coffee_X_train[i]
  image_label = coffee_y_train[i]
  plot_one_image(image, labels=[image_label])

#UsingMirror images
for i in range(500,520):
  image = mirror_X_train[i]
  image_label = mirror_y_train[i]
  plot_one_image(image, labels=[image_label])

#Attentive images
for i in range(500,520):
  image = attentive_X_train[i]
  image_label = attentive_y_train[i]
  plot_one_image(image, labels=[image_label])


### Color matrices

image = image.reshape(64, 64, 3)
new_image = image.copy()

new_image[:, :, 0] = 1 #RGB
new_image[:, :, 1] = 0
plot_one_image(new_image, labels=[image_label])

#croppiing
rect_image = image.copy()
image[5,5]
image[5,5,0] #for red color

#rectangle to maintain privacy
new_image = np.copy(image)

# experimenting rows and columns, can be modified
start_row = 10
stop_row = 28
start_col = 15
stop_col = 27

# red color
new_color = [1, 0, 0]

# Use slicing to draw a rectangle
new_image[start_row:stop_row, start_col:stop_col] = new_color

plot_one_image(new_image, labels=[image_label])


#Machine learning KNN
(X_train, y_train) = get_train_data(flatten=True)
(X_train, y_train) = get_train_data(flatten=True)
(X_test, y_test) = get_test_data(flatten=True)


knn = KNeighborsClassifier(n_neighbors = 3) #change n value and check for accuracy. I used 3 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
