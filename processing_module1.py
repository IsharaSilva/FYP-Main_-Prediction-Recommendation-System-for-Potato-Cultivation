import cv2
import numpy as np # dataAnalysis
import pandas as pd #multidimensional arrays
import matplotlib.pyplot as plt #data visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy


# Define the list of transformations
transformations = [
    ('rotation', cv2.ROTATE_90_CLOCKWISE),  # Rotate the image 90 degrees clockwise
    ('translation', np.float32([[1, 0, 50], [0, 1, 50]])),  # Translate the image by 50 pixels in both x and y directions
    ('scaling', np.float32([[0.5, 0, 0], [0, 0.5, 0]])),  # Scale down the image by 0.5
    ('flipping', 0),  # Flip the image horizontally
    ('noise', 30)  # Add Gaussian noise with mean=0 and standard deviation=30
]

x = []
y = []
features = []
labels = []

# GLCM parameters
distances = [1]
angles = [0]

# Preprocess and transform image
def preprocess_image(img):
        img1 = cv2.resize(img, (256, 256))  # Image resizing
        img1 = cv2.medianBlur(img1, 5)  # Apply median filter to remove noise
        img1 = img_as_ubyte(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # Contrast improvement

        # Convert the image to LAB color space
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)

        # Split the LAB image into separate channels
        img1_lab_planes = cv2.split(img1_lab)

        # Apply histogram equalization to the L channel
        img1_lab_planes = list(img1_lab_planes)
        img1_lab_planes[0] = cv2.equalizeHist(img1_lab_planes[0])
        img1_lab_planes = tuple(img1_lab_planes)

        # Merge the planes and convert back to RGB
        img1_lab = cv2.merge(img1_lab_planes)
        img1 = cv2.cvtColor(img1_lab, cv2.COLOR_LAB2RGB)

        # Smoothing
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img1_lab_planes = cv2.split(cv2.cvtColor(img1, cv2.COLOR_RGB2LAB))
        img1_lab_planes = list(img1_lab_planes)
        img1_lab_planes[0] = clahe.apply(img1_lab_planes[0])
        img1_lab_planes = tuple(img1_lab_planes)
        img1_lab = cv2.merge(img1_lab_planes)
        img1 = cv2.cvtColor(img1_lab, cv2.COLOR_LAB2RGB)

        # Append the preprocessed image and original image to the lists
        x.append(img1)
        y.append(img)

        # Apply transformations to the image
        for transform_name, transform_param in transformations:
            transformed_img = img1.copy()

            if transform_name == 'rotation':
                transformed_img = cv2.rotate(transformed_img, transform_param)
            elif transform_name == 'translation':
                M = transform_param
                transformed_img = cv2.warpAffine(transformed_img, M, (img1.shape[1], img1.shape[0]))
            elif transform_name == 'scaling':
                M = transform_param
                transformed_img = cv2.warpAffine(transformed_img, M, (int(img1.shape[1]*0.5), int(img1.shape[0]*0.5)))
            elif transform_name == 'flipping':
                transformed_img = cv2.flip(transformed_img, transform_param)
            elif transform_name == 'noise':
                noise = np.random.normal(0, transform_param, img1.shape).astype(np.uint8)
                transformed_img = cv2.add(transformed_img, noise)
        return transformed_img



# Perform feature extraction
def Features(img):

        # Split RGB image into channels
        blue, green, red = cv2.split(img)

        # Calculate mean and standard deviation for each channel
        mean_blue = blue.mean()
        mean_green = green.mean()
        mean_red = red.mean()
        std_dev_blue = blue.std()
        std_dev_green = green.std()
        std_dev_red = red.std()

        # Convert RGB image to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Split HSV image into channels
        hue, saturation, value = cv2.split(hsv_img)

        # Calculate mean hue
        mean_hue = hue.mean()

        # Convert RGB image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute GLCM
        glcm = greycomatrix(gray_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Compute GLCM properties
        #contrast = greycoprops(glcm, prop='contrast')[0, 0]
        #dissimilarity = greycoprops(glcm, prop='dissimilarity')[0, 0]
        #homogeneity = greycoprops(glcm, prop='homogeneity')[0, 0]
        energy = greycoprops(glcm, prop='energy')[0, 0]
        #correlation = greycoprops(glcm, prop='correlation')[0, 0]


        # Calculate ASM and entropy
        asm = (glcm**2).sum()
        entropy = shannon_entropy(gray_img)

        # Store the extracted features and labels
        feature = [
            energy,
            asm, entropy,
            mean_hue, mean_red, mean_green, mean_blue,
            std_dev_red, std_dev_green, std_dev_blue
        ]
        features.append(feature)


# Print the extracted features and labels for each image
for i, feature in enumerate(features):
    print(f"Image {i+1}: Features={feature}, Label={labels[i]}")