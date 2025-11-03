#Import TensorFlow for loading the trained model and running inference
import tensorflow as tf
import os
import numpy as np
import cv2

best_model_file = "C:/Users/nitro/OneDrive/Desktop/Model_Training_and_Evaluation/best_model_resnet.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

input_shape = (224,224)

### Base folder where your class folders (sports) are stored
path = "C:/Users/nitro/OneDrive/Desktop/Model_Training_and_Evaluation/Dataset" 

### List all category folder names from disk (must match training source)
categories = os.listdir(path)

### Sort categories to keep the same label order used in training
categories.sort()

### Verify class names visually for sanity checking
print(categories)

### Confirm the number of categories matches the model's output units
print(len(categories))


### Define a utility to resize, add batch dimension, and normalize an image
def prepareImage(img) :
    ### Resize to the model input shape with area interpolation (good for downscaling)
    resized = cv2.resize(img, input_shape , interpolation = cv2.INTER_AREA)
    ### Add a batch dimension so the array shape becomes (1, H, W, 3)
    imgResult = np.expand_dims(resized , axis= 0)
    ### Normalize pixel values to [0,1] to match training-time scaling
    imgResult = imgResult / 255.
    ### Return the preprocessed tensor
    return imgResult

# Provide the actual test image path to classify now
testImagePath = r"C:\Users\NimaLama\Downloads\dal+bhat.webp"

### Read the raw image from disk (BGR layout in OpenCV)
img = cv2.imread(testImagePath)
print(img)


### Apply your preprocessing pipeline so it matches the training distribution
ImageForModel = prepareImage(img)

### Run a forward pass through the network to obtain class probabilities
result = model.predict(ImageForModel , verbose=1)

### Collapse probabilities to the highest scoring class index
answers = np.argmax(result , axis=1)

### Print the numeric class index (useful sanity check)
print(answers)

### Convert the class index to the human-readable label via your categories list
text = categories[answers[0]]

### Print the predicted label text for the console log
print("The predicted class is : " + text)
### Choose an OpenCV font for overlaying the predicted label
font = cv2.FONT_HERSHEY_COMPLEX

### Draw the predicted text near the top-left corner in yellow with thickness 2
cv2.putText(img , text , (20,20) , font , 1, (0,255,255), 2) # Yellow color

### Open an OS-native window to preview the labeled image
cv2.imshow("img", img)

### Wait indefinitely until a key is pressed (press any key to continue)
cv2.waitKey(0)

### Close all OpenCV windows to release system resources
cv2.destroyAllWindows()
