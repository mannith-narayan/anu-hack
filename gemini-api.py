import google.generativeai as genai

# import os




import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the saved model
model_path = 'models/asbestos_detector.h5'
model = load_model(model_path)

# Define image size (should match the size used during training)
IMAGE_SIZE = (224, 224)

def predict_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize to [0,1]

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Interpret the result
    probability = prediction * 100
    classification = 'Not Asbestos' if prediction > 0.5 else 'Asbestos'
    
    return probability, classification

# Specify the path to your image
image_path = 'dataset/test/test-7.jpg'  # Replace with the actual path to your image

# Make the prediction
probability, classification = predict_image(image_path)

# Print the results
print(f"Image: {image_path}")
print(f"Probability of Asbestos: {100- probability:.2f}%")
print(f"Classification: {classification}")

genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_risk_assessment(asbestos_probability, classification):
    prompt = f"""

    You are an expert who specialises in making risk assessment plans. Imagine you work for the government, meaning
    you are incredibly strict with your assessments. You have been asked to provide a risk assessment based on
    an asbestos detection test for a building renovation:
    Probability of Asbestos: {100 - asbestos_probability:.2f}%
    Classification: {classification}

    I need you to do the following:
    1. the format of the report should be the following way. It is always a table with the following 5 columns:
        a. Preventative/risk and minimisation strategy
        b. Identified Risk 
        c. Potential Source of Risk 
        d. Actions Required 
        e. Proposed Outcomes
    
    2. Depending on the classification, provide a risk assessment for both cases (Asbestos and Not Asbestos).
        a. If there is asbestos, depending on the probability, I need you to provide a detailed risk 
           assessment plan in the table format.
        b. If there is no asbestos, mention that there is no risk of asbestos but provide a general risk, but thorough
    Format the response as a structured report.
        c. At the end of the report, always mention that more images should be taken for a more accurate assessment.
        Also, mention that the report is based on the current image and the probability of asbestos detection, and that 
        the risk assessment is subject to change based on further testing. Also mention that an expert should be consulted if needed.
    
        3. Generate the report in an html format. Make sure to include the probability and classification in the report. But 
         write them in a way that is easy to understand for a layman.
    """

    response = model.generate_content(prompt) # takes a prompt as input and returns a response
    print(response.text)

# Generate a risk assessment based on the image classification
generate_risk_assessment(probability, classification)
