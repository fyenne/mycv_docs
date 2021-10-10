```yaml
Title: CV_walkthru
author: siming yan
reference: 
	kaggle: Tensorflow/Keras/GPU for Chinese MNIST Prediction
	https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
	The Deep Learning with Keras Workshop An Interactive Approach to Understanding Deep Learning with Keras, 2nd Edition by Matthew Moocarme, Mahla Abdolahnejad, Ritesh Bhagwat
```





# CV walk through

Computer Vision: Chinese word classification.

> ```
> The Chinese characters are the following:
> * 零 - for 0  
> * 一 - for 1
> * 二 - for 2  
> * 三 - for 3  
> * 四 - for 4  
> * 五 - for 5  
> * 六 - for 6  
> * 七 - for 7  
> * 八 - for 8  
> * 九 - for 9  
> * 十 - for 10
> * 百 - for 100
> * 千 - for 1000
> * 万 - for 10 thousands
> * 亿 - for 100 millions
> ```

### data

+ a pictures data set 
+ a dataset contains pics name and label. like : pic_1, 八.



### always check NAs, check interactions between pics and labels. and pic size.

```python
# ---  NAs
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(data_df)
# --- interactions 
file_names = data_df['file']
print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
# ---  pic size.

def read_image_sizes(file_name):
    image = skimage.io.imread(IMAGE_PATH + file_name)
    return list(image.shape)
m = np.stack(data_df['file'].apply(read_image_sizes))
df = pd.DataFrame(m,columns=['w','h'])
data_df = pd.concat([data_df,df],axis=1, sort=False)

```

### Train test split: 

 ```python
 train_df, test_df = train_test_split(
     data_df, test_size=TEST_SIZE, 
     random_state=RANDOM_STATE, 
     stratify=data_df["code"].values) # stratify over labels.
 ```

### resize image 

```python
def read_image(file_name):
    image = skimage.io.imread(IMAGE_PATH + file_name)
    image = skimage.transform.resize(image, \
                                     (IMAGE_WIDTH, IMAGE_HEIGHT, 1),  # resize!
                                     mode='reflect')
    return image[:,:,:]

# one hot encoding.
def categories_encoder(dataset, var='value'):
    X = np.stack(dataset['file'].apply(read_image))
    y = pd.get_dummies(dataset[var], drop_first=False)
    return X, y
```

setting some default variables.

```
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
CONV_2D_DIM_1 = 16
CONV_2D_DIM_2 = 16
CONV_2D_DIM_3 = 32
CONV_2D_DIM_4 = 64
MAX_POOL_DIM = 2
KERNEL_SIZE = 3
BATCH_SIZE = 32
NO_EPOCHS = 5 
DROPOUT_RATIO = 0.2
PATIENCE = 5
VERBOSE = 1
```

### image Augmentation:

```python
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.0,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,
                                   horizontal_flip = False)
test_datagen = ImageDataGenerator(rescale = 1./255.0)

```

+ Reduces overfitting: It helps reduce overfitting by creating multiple versions of the 
  same image, rotated by a given amount.
+  Increases the number of images: A single image acts as multiple images. So, 
  essentially, the dataset has fewer images, but each image can be converted into 
  multiple images with image augmentation. Image augmentation will increase the 
  number of images and each image will be treated differently by the algorithm.
+ Easy to predict new images: Imagine that a single image of a football is looked at 
  from different angles and each angle is considered a distinct image. This will mean 
  that the algorithm will be more accurate at predicting new images



## build up model:

### feature detector &max pooling & flatten

![image-20210918161204311](C:\Users\dscshap3808\Documents\my_scripts_new\mycv_plays\doc_pics\image-20210918161204311.png)

![image-20210918161144858](C:\Users\dscshap3808\Documents\my_scripts_new\mycv_plays\doc_pics\image-20210918161144858.png)

![image-20210918161431613](C:\Users\dscshap3808\Documents\my_scripts_new\mycv_plays\doc_pics\image-20210918161431613.png)

An Artificial Neural Network (ANN)

```python
model=Sequential()
model.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, \
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), 
    activation='relu', padding='same'))
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, \
    activation='relu', padding='same'))
    
model.add(MaxPool2D(MAX_POOL_DIM))
model.add(Dropout(DROPOUT_RATIO))
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(Dropout(DROPOUT_RATIO))
model.add(Flatten())
model.add(Dense(y_train.columns.size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

```python
# Notes!
Conv2D (filters, kernel_size, strides=(1, 1),\
        padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, 
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)
```

+ <filters>: number of  filters that the convolutional layer will learn. feature detectors.

  ```yaml
  So, how should you choose your filter_size ? 
  First, examine your input image — is it larger than 128×128?
  If so, consider using a 5×5 or 7×7 kernel to learn larger features and then quickly reduce spatial dimensions — then start working with 3×3 kernels
  ```

+ <kernel size>: **must be an \*odd\* integer as well.** Typical values for  kernel_size  include: (1, 1) , (3, 3) , (5, 5) , (7, 7) . It’s rare to see kernel sizes larger than *7×7*.

+ ![keras_conv2d_padding](.\doc_pics\keras_conv2d_padding.gif)

+ <strides>:  The strides parameter is a 2-tuple of integers, specifying the “step” of the convolution along the *x* and *y* axis of the input volume. typically you’ll leave the strides parameter with the default (1, 1) value; however, you may occasionally increase it to (2, 2) to help reduce the size of the output volume 

  ```yaml
  A given convolutional filter is applied to the current location of the input volume;
  The filter takes a 1-pixel step to the right and again the filter is applied to the input volume
  ```

+ <Padding>: The padding parameter to the Keras Conv2D class can take on one of two values: valid or same.

  ```yaml
  Valid: the spatial dimensions are allowed to reduce via the natural application of convolution 
  Same: If you instead want to preserve the spatial dimensions of the volume such that the output volume size matches the input volume size.
  
  set it to same for the majority of the layers in my network and then either reduce spatial dimensions of my volume by either
  + Max pooling
  + Strided convolution
  ```

+ <activation>: relu relu relu.... softmax..

### tune and train

```python
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)
checkpointer = ModelCheckpoint('best_model.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)


%%time
train_model  = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[earlystopper, checkpointer, annealer])
```

### test , per class results:

```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted)) 
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
    
test_accuracy_report(model)

```

### prediction with best model:

```python
model_optimal = model
model_optimal.load_weights('best_model.h5')
score = model_optimal.evaluate(X_test, y_test, verbose=0)
print(f'Best validation loss: {score[0]}, accuracy: {score[1]}')

test_accuracy_report(model_optimal)
```





### loss error and confusion matrix plot 

```python
# Thanks to https://www.kaggle.com/vbmokin/tensorflow-keras-gpu-for-chinese-mnist-prediction
def create_trace(x,y,ylabel,color):
        trace = go.Scatter(
            x = x,y = y,
            name=ylabel,
            marker=dict(color=color),
            mode = "markers+lines",
            text=x
        )
        return trace
    
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1,len(acc)+1))
    #define the traces
    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")
    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")
    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")
    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',
                                                             'Training and validation loss'))
    #add traces to the figure
    fig.append_trace(trace_ta,1,1)
    fig.append_trace(trace_va,1,1)
    fig.append_trace(trace_tl,1,2)
    fig.append_trace(trace_vl,1,2)
    #set the layout for the figure
    fig['layout']['xaxis'].update(title = 'Epoch')
    fig['layout']['xaxis2'].update(title = 'Epoch')
    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])
    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])
    #plot
    iplot(fig, filename='accuracy-loss')
```

```python
def plot_cm(train, target_train):
# Look at confusion matrix 
# Thanks to https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Predict the values from the validation dataset
    Y_pred = model.predict(train)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(target_train,axis = 1) 
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 
    
plot_cm(X_train, Y_train)
plot_cm(X_test, Y_test)
```



### outliers analysis

```python
def pred_wrong_display_MNIST_dataset(X_test, predictions, Y_test):
    # Displays misclassified digits from MNIST dataset
    
    X_test_wrong = []
    predictions_wrong = []
    Y_test_pred = []
    for i in range(len(X_test)):
        Y_test_pred.append(np.argmax(Y_test[i]))
        if predictions[i] != Y_test_pred[i]:
            #print(i, predictions[i], Y_test_pred[i])
            X_test_wrong.append(X_test[i])
            predictions_wrong.append(predictions[i])

    plot_images_sample(X_test_wrong, predictions_wrong)
        
    print('Accuracy is', round(accuracy_score(Y_test_pred, predictions),3))
    
    return Y_test_pred


# Displays misclassified digits from MNIST
Y_test_pred = pred_wrong_display_MNIST_dataset(X_test, predictions, Y_test)
```

---

<footer>
## Transfer learning & pre-trained models:

###  fine tuning:

There is a three-point system to working with fine-tuning: 

1. Add a classifier (ANN) on top of a pre-trained system. 
2. Freeze the convolutional base and train the network.
3. Train the added classifier and the unfrozen part of the convolutional  base jointly.

### ImageNet data set 

+ VGG16  `16 layers`
+ Inception V3  
+ Xception 
+ ResNet50 `50 layers`
+ MobileNet 

```python
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# load model
classifier = VGG16()
# load new pics
new_image = image.load_img('../Data/Prediction/pizza.jpg.jpg', target_size=(224, 224))

# transform new pics
transformed_image = image.img_to_array(new_image)
transformed_image = np.expand_dims(transformed_image, axis=0)
    	# transformed_image.shape ->(1,244,244,3)
transformed_image = preprocess_input(transformed_image)
    	# transformed_image.shape must be (1, 244,244,3)

# predict new pics
y_pred = classifier.predict(transformed_image)
from keras.applications.vgg16 import decode_predictions
decode_predictions(y_pred,top=5)

# print probobility 
label = decode_predictions(y_pred)
# Most likely result is retrieved, for example, the highest  probability
decoded_label = label[0][0]
# The classification is printed 
print('%s (%.2f%%)' % (decoded_label[1], decoded_label[2]*100 ))
```

### fine tune vgg-16 (create a similar model actually)

#### Remove the last layer, labeled predictions in the preceding image, from the  model summary. 

Create a new Keras model of the sequential class and iterate  through all the layers of the VGG model. Add all of them to the new model, except  for the last layer

```python
vgg_model = keras.applications.vgg16.VGG16()
vgg_model.summary() 

last_layer = str(vgg_model.layers[-1])
np.random.seed(42)
random.set_seed(42)
classifier= keras.Sequential()
for layer in vgg_model.layers:
    if str(layer) != last_layer:
        classifier.add(layer)
```

#### freeze initial layers & add new classification layer

```python
for layer in classifier.layers:
    layer.trainable=False
    
classifier.add(Dense(1, activation='sigmoid'))
classifier.summary()

classifier.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])
```

````python
from keras.preprocessing.image import ImageDataGenerator
generate_train_data = ImageDataGenerator(rescale = 1./255,
                                         shear_range = 0.2, 
                                         zoom_range = 0.2,
                                         horizontal_flip = True)
generate_test_data = ImageDataGenerator(rescale =1./255)
training_dataset = generate_train_data.flow_from_directory(
    '../Data/dataset/training_set',target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary')
test_datasetset = generate_test_data.flow_from_directory(
    '../Data/dataset/test_set',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary')
classifier.fit_generator(training_dataset,
                         steps_per_epoch = 100,
                         epochs = 10,
                         validation_data = test_datasetset,
                         validation_steps = 30,
                         shuffle=False)
````

````python
from keras.preprocessing import image
new_image = image.load_img('../Data/Prediction/test_image_2.jpg', 
                           target_size = (224, 224))
new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis = 0)
result = classifier.predict(new_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'It is a flower'
else:
    prediction = 'It is a car'
print(prediction)

````

