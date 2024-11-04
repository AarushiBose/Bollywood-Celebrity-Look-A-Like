# Bollywood-Celebrity-Look-A-Like (Know Your Bollywood Twin!)

## Description
Bollywood Twin is a facial recognition project that identifies Bollywood actors who resemble a given uploaded image. Utilizing deep learning models for feature extraction, the application matches the facial features of the uploaded image with a database of Bollywood actors.

## Dataset Used
Link - https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset

This dataset contains localized faces of 100 Bollywood celebrities. Each class includes between 80 to 150 samples, with each image sized at 64 x 64 pixels. The samples encompass various wild conditions, including different orientations, lighting conditions, and age transitions.

## Features
### ➢ Face Detection and Feature Extraction
The project employs the VGGFace model, a powerful convolutional neural network (CNN) specifically designed for face recognition tasks. The model is capable of extracting deep facial features from images, which are critical for accurately identifying and matching faces. By utilizing pre-trained weights on a large dataset, the VGGFace model significantly enhances the accuracy of feature extraction, allowing the application to recognize various nuances in facial characteristics, even under different conditions.

### ➢ Cosine Similarity for Facial Matching
To determine the resemblance between the uploaded image and the database of Bollywood actors, the project uses cosine similarity. This metric measures the cosine of the angle between two non-zero vectors in an inner product space, providing a measure of similarity. By comparing the extracted feature vectors of the uploaded image against the feature vectors of the actors, the application identifies the closest match. This approach is robust and efficient, allowing for quick comparisons and accurate recommendations.

### ➢ User-Friendly Interface with Streamlit
The application features an intuitive user interface built using Streamlit, which simplifies the process of uploading images and displaying results. Users can easily interact with the application by uploading their photos directly through the web interface. Once an image is uploaded, the application processes it and displays the most similar Bollywood actor alongside the uploaded image. The design prioritizes ease of use, ensuring that even those with minimal technical knowledge can navigate and utilize the application effectively.

### ➢ Real-Time Recommendations
The system is designed for real-time processing, enabling users to receive immediate feedback on their uploaded images. The backend processes the image, extracts features, calculates similarities, and presents the results in a matter of seconds, enhancing the user experience and engagement.

### ➢ Versatile Use Cases
Beyond entertainment, the application has potential use cases in social media filters, mobile applications, and as a fun tool for events where users can discover their Bollywood look-alikes. The underlying technology can be further adapted for other celebrity databases, making it a flexible solution for face recognition applications across various domains.

## Demonstrative Overview

![WhatsApp Image 2024-10-31 at 23 37 58_43a13e4d](https://github.com/user-attachments/assets/15348cb2-4075-4a6e-a27d-d46e77ecad16)

![image](https://github.com/user-attachments/assets/51602c10-c4da-4995-9db3-a62ae4fe51a2)

![image](https://github.com/user-attachments/assets/599208ab-c9d7-4a58-8fa7-89fed5ebaba7)

![image](https://github.com/user-attachments/assets/d462ffd6-ac86-47b4-af4e-e275628efab2)


