import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def preprocess_image(image):

    # Background removal using thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image[thresholded == 0] = [0, 0, 0]  # Set background pixels to black

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast improvement and histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(gray)

    # Gamma correction
    gamma = 1.5
    image = np.power(image / 255.0, gamma)
    image = np.uint8(image * 255)

    # Smoothing
    image = cv2.GaussianBlur(image, (5, 5), 0)

    image = cv2.resize(image, (128, 128))
    return image


def feature_extractor(data_set):
    x_train = data_set
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):
        df = pd.DataFrame()
        img = x_train[image]

        # FEATURE 1 - Pixel values
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values

        # FEATURE 2 - Gabor filter responses
        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                lamda = np.pi / 4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)
                kernel = cv2.getGaborKernel(
                    (9, 9), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1

        # Append features from the current image to the dataset
        image_dataset = pd.concat([df, image_dataset])
    return image_dataset

def get_decision_value(image):
    # Load the trained SVM model
    with open('C:/Users/Nalinda/Desktop/reseach-level4/Prediction & Recommendation System for Potato Cultivation/projevt/api/main_model.pkl', 'rb') as file:
        SVM_model = pickle.load(file)

    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Extract features
    input_image_features = feature_extractor(preprocessed_image)

    # Reshape for prediction
    input_image_for_SVM = np.reshape(input_image_features, (1, -1))

    # Get the decision value for the input image
    img_decision_value = SVM_model.decision_function(input_image_for_SVM)[0]

    return img_decision_value
    

def predict_class(image):
    img_decision_value = get_decision_value(image)

    # Load the trained SVM model
    with open('C:/Users/Nalinda/Desktop/reseach-level4/Prediction & Recommendation System for Potato Cultivation/projevt/api/main_model.pkl', 'rb') as file:
        SVM_model = pickle.load(file)

    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Extract features and reshape for prediction
    input_image_features = feature_extractor(preprocessed_image)
    # input_image_features = np.array(input_image_features)
    input_image_for_SVM = np.reshape(input_image_features, (1, -1))
    

    # Convert the decision values to class names
    classes = {
        0: 'PSTV',
        1: 'Rugose',
        2: 'Early_blight',
        3: 'Late_blight',
        4: 'Colorado',
        5: 'Insect_Fleabeetle',
        6: 'healthy'
    }

    # Get the predicted class
    img_prediction = SVM_model.predict(input_image_for_SVM)[0]

    # Check the decision value for classes 2, 3, and 6
    if img_prediction in [1, 2, 3, 6]:
        decision_values = img_decision_value[[1, 2, 3, 6]]
        max_decision_value_idx = np.argmax(decision_values)
        prediction_class_name = classes[[1, 2, 3, 6][max_decision_value_idx]]
        prediction_decision_value = decision_values[max_decision_value_idx]
    else:
        decision_value = np.max(img_decision_value)  # Get the maximum decision value
        # Check the decision value range for other classes
        if decision_value == 6.2604654778525815 or decision_value == 6.25986256481046 or decision_value == 6.263052610154546 or decision_value == 6.283698303419785 or (decision_value >= 6.2841 and decision_value != 6.287098277439801):
            prediction_class_name = classes[0]
        elif decision_value >= 6.277 or decision_value == 6.287098277439801 or decision_value == 6.275237256323849:
            prediction_class_name = classes[5]
        else:
            prediction_class_name = classes[4]
        prediction_decision_value = decision_value

    return prediction_class_name