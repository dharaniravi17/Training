import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
first_conv_layer = model.layers[0]
filters, biases = first_conv_layer.get_weights()

print(f"Filters shape: {filters.shape}") 
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters = 6 
fig, axs = plt.subplots(1, n_filters, figsize=(20,5))

for i in range(n_filters):
    f = filters[:, :, :, i]
    axs[i].imshow(f, interpolation='nearest')
    axs[i].set_title(f'Filter {i+1}')
    axs[i].axis('off')

plt.suptitle('Visualization of First Layer Filters')
plt.show()
test_image = X_test[0]
test_image_input = np.expand_dims(test_image, axis=0)  

activation_model = models.Model(inputs=model.input, outputs=model.layers[0].output)
feature_maps = activation_model.predict(test_image_input)

print(f"Feature maps shape: {feature_maps.shape}") 
square = 6  
fig, axs = plt.subplots(square, square, figsize=(12,12))

for i in range(square * square):
    ax = axs[i//square, i%square]
    ax.imshow(feature_maps[0, :, :, i], cmap='gray')
    ax.axis('off')

plt.suptitle('Feature Maps after First Conv Layer')
plt.show()

print("\nInterpretation:")
print(" The filters tend to detect basic patterns such as edges, color gradients, and simple textures.")
print(" The feature maps highlight the parts of the input image where these patterns occur.")
