import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import pytesseract as pt
import os

def create_model():
    inception_resnet = InceptionResNetV2(weights=None, include_top=False,
                                       input_tensor=Input(shape=(224,224,3)))
    headmodel = inception_resnet.output
    headmodel = Flatten()(headmodel)
    headmodel = Dense(500,activation="relu")(headmodel)
    headmodel = Dense(250,activation="relu")(headmodel)
    headmodel = Dense(4,activation='sigmoid')(headmodel)
    return Model(inputs=inception_resnet.input, outputs=headmodel)

# Initialize model variable
model = None

# Create and load the model
model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'object_detection.h5')
if not os.path.exists(model_path):
    print(f"Model file does not exist at path: {model_path}")
else:
    try:
        # Create model with same architecture as training
        model = create_model()
        # Load the weights
        model.load_weights(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)

def object_detection(path,filename):
    global model  # Declare model as global to access it
    if model is None:
        raise Exception("Model is not loaded. Please check the model file.")
    # read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    # data preprocessing
    image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    # make predictions
    coords = model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    # convert into bgr
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),image_bgr)
    return coords

def preprocess_license_plate(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Resize image to a larger size for better OCR
    scale_factor = 4  # Increased scale factor further
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Noise reduction
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Dilation to connect components
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    return thresh

def OCR(path,filename):
    try:
        # Check if tesseract is installed and set path for Windows
        if os.name == 'nt':
            pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        img = np.array(load_img(path))
        cods = object_detection(path,filename)
        xmin ,xmax,ymin,ymax = cods[0]
        
        # Add padding to ROI
        padding = 20
        ymin = max(0, ymin - padding)
        ymax = min(img.shape[0], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(img.shape[1], xmax + padding)
        
        roi = img[ymin:ymax,xmin:xmax]
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        
        # Save original ROI
        cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
        
        # Preprocess the image
        processed_roi = preprocess_license_plate(roi_bgr)
        
        # Save processed ROI for debugging
        cv2.imwrite('./static/roi/processed_{}'.format(filename), processed_roi)
        
        # Custom Tesseract configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        
        # Get text using tesseract
        text = pt.image_to_string(processed_roi, config=custom_config)
        
        # Clean and format the text
        text = text.strip()
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Keep only alphanumeric characters
            cleaned = ''.join(c for c in word if c.isalnum())
            if cleaned:
                cleaned_words.append(cleaned)
        
        # Format for Indian license plate
        if len(cleaned_words) >= 4:
            # Ensure first part is 2 letters (state code)
            if len(cleaned_words[0]) >= 2 and cleaned_words[0][:2].isalpha():
                state_code = cleaned_words[0][:2]
                # Ensure second part is 2 digits (district code)
                if len(cleaned_words[1]) >= 2 and cleaned_words[1][:2].isdigit():
                    district_code = cleaned_words[1][:2]
                    # Get the series and number
                    series = cleaned_words[2] if len(cleaned_words) > 2 else ""
                    number = cleaned_words[3] if len(cleaned_words) > 3 else ""
                    
                    final_text = f"{state_code} {district_code} {series} {number}"
                    print("Extracted text:", final_text)
                    return final_text
        
        # If format doesn't match, return original cleaned text
        final_text = ' '.join(cleaned_words)
        print("Extracted text:", final_text)
        return final_text if final_text else "No text detected"
        
    except Exception as e:
        print(f"Error in OCR: {str(e)}")
        return "Error: Could not read text"