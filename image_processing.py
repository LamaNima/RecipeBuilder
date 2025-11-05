# import necessary modules
import numpy as np 
import cv2 
import os 

# Defining the shape of all input images
input_shape = (224,224)
path = r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\Dataset" 

print(f"Numpy: {np.__version__}")
# listing all the categories
categories = os.listdir(path)
categories.sort()

print(categories)
print(len(categories))

# Function to prepare all images and labels
def prepareData(path):
    Images = []
    Labels = []

    for category in categories:
        fullPath = os.path.join(path,category)
        file_names = os.listdir(fullPath)

        # Loop over each file in the folder
        for file in file_names:
            file = os.path.join(fullPath, file)

            img = cv2.imread(file)

            #Check if image is valid
            if img is not None:
                # Resize the image to (224,224)
                image = cv2.resize(img , input_shape, interpolation = cv2.INTER_AREA)

                # Append resized image and its category label
                Images.append(image)
                Labels.append(category)

    # Convert list of images and labels to NumPy array
    Images = np.array(Images)           
    Labels = np.array(Labels)

    # Return images and labels
    return Images , Labels

allImages , allLables = prepareData(path)

print(allImages.shape)
print(allLables.shape)

# # Displaying two sample images with their labels
# img = allImages[120]
# label = allLables[120]

# print(label)

# cv2.imshow("img1", img)

# cv2.waitKey(0)

# Saving processed data into .npy format
print("Save the data .......")
np.save(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\images_densenet.npy", allImages)
np.save(r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\labels_densenet.npy", allLables)
print("Finish save the data .......")


# Path to your main dataset folder
dataset_path = r"C:\Users\NimaLama\OneDrive\Desktop\Model_Training_and_Evaluation\Dataset"

# # List all class folders
# categories = sorted(os.listdir(dataset_path))

# print("📂 Dataset Summary:\n")

# total_images = 0

# for category in categories:
#     category_path = os.path.join(dataset_path, category)

#     # Count only files (ignore nested folders, if any)
#     num_images = len([
#         f for f in os.listdir(category_path)
#         if os.path.isfile(os.path.join(category_path, f))
#     ])

#     total_images += num_images
#     print(f"{category:20s} ➜ {num_images} images")

# print("\nTotal images in dataset:", total_images)
