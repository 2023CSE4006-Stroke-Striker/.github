# Stroke Striker

## Team

- Lee Seungsu, mqm0051@gmail.com

- Park Geonryul, geonryul0131@gmail.com

- Elia Ayoub, elia-ayoub@outlook.com

- Ryan Jabbour, jabbourryan2@gmail.com

## Table of contents

|    | Section                                       |
|---:|:----------------------------------------------|
| I  | [Introduction](#i-introduction)               |
| II | [Datasets](#ii-datasets)                      |
| III| [Methodology](#iii-methodology)               |
| IV | [Evaluation & Analysis](#iv-evaluation--analysis) |
| V  | [Related Work](#v-related-work)               |
| VI | [Conclusion](#vi-conclusion)                  |



## I. Introduction

Health is one of the most important factors in a person’s life. The increase in life expectancy over the years, due to the development of technology and healthcare, has made precaution measures against diseases much harder to take.

Stroke is a worldwide acute and severe disease, ranked as the fourth cause of death in South Korea and the fifth cause of death in the United States of America.

There are cases where the elderly are reluctant to go to the hospital due to their habits or due to their misunderstandings arising from their experiences.

However, this problem is not only defined to the older generation. A growing number of people have started living alone and because of that, they aren’t able to point out their unhealthy habits and if they get an acute disease, they won’t be able to take the proper measures to take care of themselves.

The best time to arrive at the hospital after the occurrence of a stroke is within one hour, which is a very short period of time. Fortunately, stroke has a reliable pre-hospital diagnostic method called BE-FAST (Balance, Eyes, Face, Arm, Speech, Terrible headache) that analyzes facial expression changes caused by paralysis of facial muscles; this is an obvious symptom of stroke to detect.

In this context, we thought of designing a preemptive and active health care service: a system that periodically checks the health of the people living in a household, both single-person households and old people homes, in order to detect signs of diseases in advance.

Research could be conducted afterwards using AI to study those facial expressions that correlate to certain diseases, so that in the future, people could relate those early factors to known illnesses.

If this health check service was supported by every home appliance, we could imagine a household that actively protects our health on a daily basis and not just any ordinary household that passively neglects dangerous health issues.

## II. Datasets

To develop and train our model, we used a dataset from Kaggle. This dataset contains 5029 images categorized into two classes. One class represents individuals diagnosed with acute stroke, while the other class represents individuals without such a diagnosis. 

To improve the model's accuracy, data augmentation techniques such as image flipping, rotation, and scaling were applied. These techniques contribute to creating a varied and resilient dataset, more reflective of real-world scenarios.

The dataset provides a large and diverse collection of images for training machine learning models to detect and diagnose strokes in patients. 

The augmentation techniques employed ensure that the model is exposed to a wide range of scenarios, improving its ability to generalize and make accurate predictions in real-world situations.

## III. Methodology

In order to create our stroke detector using an artificial intelligence model, we explored the capabilities of three machine learning algorithms—Random Forest, SVM, and ViT—each presenting unique methodologies and insights.

- ### Random Forest

    The first algorithm that we used was the Random Forest machine learning model in order to train our AI model. The process is divided into three parts. Firstly, loading and pre-processing face data of stroke patients and normal people. Secondly, loading and training the random forest model on the preprocessed data. Lastly, testing the trained model. 

    Data preprocessing consists of loading an image from a folder and converting it to black and white. Since the color and expression of the image are independent, we proceed to reduce the dimension of the input. Since the size of the image data is different, we convert it to a size of (150, 150) and flatten it into a vector. The model is imported using scikit-learn’s library. The training data and testing data were divided into 80% and 20% and learned using the training data. This process took only a few minutes. Finally, the trained model is verified with testing data.

    At this time, the performance of the stroke judgment model cannot be assured. Since it is a judgment of a person’s health, it must be precise, taking into consideration true positives and false negatives. To determine more accurate performances, we used the classification report function. What is important to note is that the stroke precision value is very high at 0.97; this means that if the model determines that the data indicates a stroke, there is a 97% probability that it is actually a stroke. On the other hand, stroke recall drops to 0.72, which means that 28% of cases were actually strokes, but the model failed to judge them as strokes.

- ### SVM (Support Vector Machine)

    The second algorithm that we used was to conduct learning based on Support Vector Machine. It is the same step-by-step process used in the Random Forest algorithm: data preprocessing, learning, and verification.

    For data preprocessing, the color of the image was converted to black and white, the number of dimensions of the data was reduced, and the size was batch converted to (150, 150). The difference from the Random Forest algorithm in the image preprocessing process was that the number of training data is limited to 500. When the training was performed using all the data with Google Colab CPU, training did not end even after more than 5 hours. We judged that it would be unreasonable to use all the data for training, so we limited the number of data to 500.

    Then, we loaded scikit-learn’s SVC and ran training. The training time, which consists of 80% training data and 20% testing data, is approximately 1 hour.

    We used classification report to obtain detailed accuracy and obtained an accuracy of 88% in all cases. However, the amount of test data was very small, so it was difficult to interpret the results as meaningful, and learning was also difficult, so we started looking for a different model.

    We saved the trained model using the joblib function, moved it to AWS Lightsail, and deployed it to a web server. At this time, the size of the model exceeded 200MB. This is also another reason we started looking for other models. 

## IV. Evaluation & Analysis

## V. Related Work

1.	Project MONAI

MONAI is an initiative started by NVIDIA and King's College London to establish an inclusive community of AI researchers to develop and exchange best practices for AI in healthcare. This collaboration has expanded to include academic and industry leaders throughout the medical field.

This project is similar to our project because it simply analyzes MRI and CT photographs with AI, but the methods used are different.

2.	BASLER

This company actually provides an overall solution for the vision system. Their products support hardware and software at the same time and can analyze images based on machine learning. However, their cameras and sensors are very expensive, so it would be difficult to apply them to home appliances as we presented them in our project.

3.	Kaggle Project

This is a stroke detection project undertaken by Kaggle. It could be used as an AI model for our project but since the algorithm used in this project is based on 2D images, it differs from the 3D recognition we need to use in our project.

4.	Multi-Angle detector

Reference: Han Gao, Amir Ali Mokhtarzadeh, Shaofan Li, Hongyan Fei, Junzuo Geng, and Deye Wang. Multi-angle face expression recognition based on integration of lightweight deep network and key point feature positioning. Journal of Physics: Conference Series, 2467, 2023.

This paper introduces lightweight deep network and combining key point feature positioning for multi-angle facial expression recognition. Using robot dog to recognize facial expressions will be affected by distance and angle. To solve this problem, this paper proposes a method for facial expression recognition at different distances and angles, which solved the larger distance and deflection angle of facial expression recognition accuracy and real-time issues.

5.	Raspberry Pi Based Emotion Recognition using OpenCV, TensorFlow, and Keras

Reference: JOYDIP DUTTA. https://circuitdigest.com/microcontroller-projects/raspberry-pi-based-emotion-recognition-using-opencv-tensorflow-and-keras.

In this tutorial, they implement an Emotion Recognition System or a Facial Expression Recognition System on a Raspberry Pi 4. They apply a pre-trained model in order to recognize the facial expression of a person from a real-time video stream. The “FER2013” dataset is used to train the model with the help of a VGG-like Convolutional Neural Network (CNN).

6.	Connect a Raspberry Pi or other device with AWS

Reference: AWS. https://docs.aws.amazon.com/iot/latest/developerguide/connecting-to-existing-device.html.

This step-by-step tutorial guides you through all the steps you need to take in order to connect a Raspberry Pi or any other device with AWS. It explains to you how to set up the device, install the required tools and libraries for the AWS IoT Device SDK, install AWS IoT Device SDK, install and run the sample app, as well as view the messages from the sample app in the AWS IoT console.

7.	Realtime Facial Emotion Recognition

Reference: victor369basu. https://github.com/victor369basu/facial-emotion-recognition.

This repository demonstrates an end-to-end pipeline for real-time Facial emotion recognition application through full-stack development. The front end is developed in react.js and the back end is developed in FastAPI. The emotion prediction model is built with Tensorflow Keras, and for real-time face detection with animation on the front-end, Tensorflow.js has been used.

8.	Kaggle FER-2013 DataSet

Reference: MANAS SAMBARE. https://www.kaggle.com/datasets/msambare/fer2013.

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.
The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

9.	Facial landmarks with dlib, OpenCV, and Python

Reference: Adrian Rosebrock. https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/.

This post explains line by line the source code provided and demonstrates in detail what are facial landmarks and how to detect them using dlib, OpenCV, and Python. Also, it introduces alternative facial landmark detectors such as ones coming from the MediaPipe library which is capable of computing a 3D face mesh.

## VI. Conclusion 
