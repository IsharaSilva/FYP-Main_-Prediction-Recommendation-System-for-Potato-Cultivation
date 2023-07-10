import numpy as np
from skimage import img_as_ubyte
import cv2

# Preprocess image
def preprocess_image(image):
    # Resize the image
    resized_image = cv2.resize(image, (400, 400))

    # Convert to RGB and apply contrast enhancement
    color_contrast_image = img_as_ubyte(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Apply Gaussian blur for smoothing
    smooth_image = img_as_ubyte(cv2.GaussianBlur(color_contrast_image, (5, 5), 0, borderType=cv2.BORDER_CONSTANT))

    # Split channels for histogram equalization
    R, G, B = cv2.split(smooth_image)

    # Apply histogram equalization
    op_R = cv2.equalizeHist(R)
    op_G = cv2.equalizeHist(G)
    op_B = cv2.equalizeHist(B)
    histogram_image = cv2.merge((op_R, op_G, op_B))

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    op2_R = clahe.apply(R)
    op2_G = clahe.apply(G)
    op2_B = clahe.apply(B)
    clahe_image = cv2.merge((op2_R, op2_G, op2_B))

    return clahe_image

# Perform image segmentation
def segment_image(image):
    # Canny Edge Detection
    edges = cv2.Canny(image=image, threshold1=10, threshold2=20)
    edge_detect_image = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

    # Thresholding segmentation method
    gray = cv2.cvtColor(edge_detect_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Create a green mask with the same size as the image
    mask = np.full_like(thresh, 255, dtype=np.uint8)

    # Apply the mask to the image
    result = cv2.bitwise_and(thresh, mask)

    return result