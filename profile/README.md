# Stroke Striker

## Demo Video
[![Alt text](https://img.youtube.com/vi/XglI0ev_tus&t=20.jpg)](https://www.youtube.com/watch?v=XglI0ev_tus&t=20)


## PPT file

Check out the project documentation in the `Documentation` folder:

- [Download PPT](Strike-Stroke-ppt-Ver4.1.key)

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
| III| [Methodology & Performance Analysis](#iii-methodology--performance-analysis)               |
| IV  | [Related Work](#iv-related-work)               |
| V | [Conclusion](#v-conclusion)                  |



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

## III. Methodology & Performance Analysis

In order to create our stroke detector using an artificial intelligence model, we explored the capabilities of three machine learning algorithms—Random Forest, SVM, and ViT—each presenting unique methodologies and insights.

- ### Random Forest

    The first algorithm that we used was the Random Forest machine learning model in order to train our AI model. The process is divided into three parts. Firstly, loading and pre-processing face data of stroke patients and normal people. Secondly, loading and training the random forest model on the preprocessed data. Lastly, testing the trained model. 

    Data preprocessing consists of loading an image from a folder and converting it to black and white. Since the color and expression of the image are independent, we proceed to reduce the dimension of the input. Since the size of the image data is different, we convert it to a size of (150, 150) and flatten it into a vector. The model is imported using scikit-learn’s library. The training data and testing data were divided into 80% and 20% and learned using the training data. This process took only a few minutes. Finally, the trained model is verified with testing data.

    At this time, the performance of the stroke judgment model cannot be assured. Since it is a judgment of a person’s health, it must be precise, taking into consideration true positives and false negatives. To determine more accurate performances, we used the classification report function. What is important to note is that the stroke precision value is very high at 0.97; this means that if the model determines that the data indicates a stroke, there is a 97% probability that it is actually a stroke. On the other hand, stroke recall drops to 0.72, which means that 28% of cases were actually strokes, but the model failed to judge them as strokes.

    ```python
    Categories=['noStroke','stroke']
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir='/content/drive/MyDrive/main'

    for i in Categories:
        print(f'loading...category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            image = io.imread(os.path.join(path, img))
            img_gray = color.rgb2gray(image)
            img_resized = transform.resize(img_gray, (150, 150))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    ```
    *Pre-processing data in Random Forest*

    <img width="700" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/63232228-b676-4ab2-9525-f82cc7e81cc8">
  
    *The result of testing Random Forest model*

- ### SVM (Support Vector Machine)

    The second algorithm that we used was to conduct learning based on Support Vector Machine. It is the same step-by-step process used in the Random Forest algorithm: data preprocessing, learning, and verification.

    For data preprocessing, the color of the image was converted to black and white, the number of dimensions of the data was reduced, and the size was batch converted to (150, 150). The difference from the Random Forest algorithm in the image preprocessing process was that the number of training data is limited to 500. When the training was performed using all the data with Google Colab CPU, training did not end even after more than 5 hours. We judged that it would be unreasonable to use all the data for training, so we limited the number of data to 500.

    Then, we loaded scikit-learn’s SVC and ran training. The training time, which consists of 80% training data and 20% testing data, is approximately 1 hour.

    We used classification report to obtain detailed accuracy and obtained an accuracy of 88% in all cases. However, the amount of test data was very small, so it was difficult to interpret the results as meaningful, and learning was also difficult, so we started looking for a different model.

    We saved the trained model using the joblib function, moved it to AWS Lightsail, and deployed it to a web server. At this time, the size of the model exceeded 200MB. This is also another reason we started looking for other models.

    ```python
    Categories=['noStroke','stroke']
    flat_data_arr=[] #input array
    target_arr=[] #output array
    num = 0
    datadir='/content/drive/MyDrive/main'
    #path which contains all the categories of images
    for i in Categories:
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            if num >= 250:
                break
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
            num = num + 1

        print(f'loaded category:{i} successfully')
        num = 0
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    ```
      *Pre-processing data in SVM model*
  
    <img width="600" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/de2414e4-46f6-4081-9b4c-214275c06fcf">
  
    *The result of testing SVM model*


- ### ViT (Vision Transformer)

    The third algorithm we used was ViT which consists of several modules. The patchify module that flattens the image for recognition, the positional embedding module that implants the positional information from the original image into the patchified data, and the MSA module that performs multi-head self-attention, the most important task in ViT. These modules are assembled into a ViT class.

    Since ViT was implemented based on pytorch, we directly implemented Dataset and Dataloader that can iterate on it. For consistency, we converted the images to black and white and resized them to (150, 150). The functions used are different, but the context is the same as the machine learning-based model above.

    The detailed implementation of the most important MSA modules went as follows. First, we declare Q, K, and V matrices as Linear in pytorch as many as the number of heads to enable learning through back propagation. Then, the Q, K, and V dot products are performed on pre-given inputs to produce a result. These results are stacked in a stack format, and when all operations are completed, they are merged into the same dimension as the input and output. In other words, during the calculation process, the division is done by the number of heads and the calculation is carried out in parallel.

    After completing the implementation, testing was conducted. As a result of training with 5 epochs using Adam optimizer, the accuracy was about 66%. It showed the lowest accuracy; this can be inferred that sufficient learning has not occurred. ViT requires a lot of data, but it is difficult to obtain a sufficient amount of learning data because it is deeply related to patients’ medical information. Additionally, there is duplicated data as well as a lot of augmented data.

    <img width="700" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/dc61248f-06d4-4f76-ab4f-e0f0dd736d33">
  
    *The result of testing ViT model*
   
    
After using all those training algorithms, we debated at the end that using Amazon Rekognition, a service for automatically training our AI model, would be an easier and more accurate choice to make.

- ### Amazon Rekognition
1) **Preparing data**:

     The data needed for learning was the same as before. Amazon Rekogniton can automatically set the name of the folder containing the data as the label of the image. Using this function, we conveniently completed labeling the data and divided the training data and testing data in a ratio of 8:2. No work was done to change the color of the image to black and white or to unify the size of the image. 

    <img width="400" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/7f969417-38e3-41c6-9828-df2bbc5eb451">
   
     *Dataset for training Amazon Rekognition*
     
3) **Training Rekognition model**: 

    Next step was training the Amazon Rekognition model with the data prepared above. Hyperparameters and those that need to be set additionally are automatically set and the optimal parameters are automatically found, so model training was performed immediately without setting the optimizer or parameters.

    <img width="700" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/80d7d874-4c13-4004-ab61-c86d5c9e88f1">
   
    *Training Amazon Rekognition automatically*

5) **Testing and deploying trained model**: 

    Once training of the model is complete, you can test it and see the results of the performance of the learned model before deployment. All results were accurately classified on the test data; the F1 score obtained was 1. Also, the confidence level of each data was quite high, showing that the model was trained very well.

    <img width="700" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/64c26fc9-b928-4d9d-8384-0d4706f7cd76">
   
    *Result of testing trained Rekognition model*
   
    <img width="700" alt="image" src="https://github.com/2023CSE4006-Stroke-Striker/.github/assets/116923946/6ae8d92e-42b3-4eb9-9ec0-f56f51d0ba35">
   
    *Deploying trained Rekognition model*


## IV. Related Work

1.	**Project MONAI**

MONAI is an initiative started by NVIDIA and King's College London to establish an inclusive community of AI researchers to develop and exchange best practices for AI in healthcare. This collaboration has expanded to include academic and industry leaders throughout the medical field.

This project is similar to our project because it simply analyzes MRI and CT photographs with AI, but the methods used are different.

2.	**BASLER**

This company actually provides an overall solution for the vision system. Their products support hardware and software at the same time and can analyze images based on machine learning. However, their cameras and sensors are very expensive, so it would be difficult to apply them to home appliances as we presented them in our project.

3.	**Kaggle Project**

This is a stroke detection project undertaken by Kaggle. It could be used as an AI model for our project but since the algorithm used in this project is based on 2D images, it differs from the 3D recognition we need to use in our project.

4.	**Multi-Angle detector**

Reference: Han Gao, Amir Ali Mokhtarzadeh, Shaofan Li, Hongyan Fei, Junzuo Geng, and Deye Wang. Multi-angle face expression recognition based on integration of lightweight deep network and key point feature positioning. Journal of Physics: Conference Series, 2467, 2023.

This paper introduces lightweight deep network and combining key point feature positioning for multi-angle facial expression recognition. Using robot dog to recognize facial expressions will be affected by distance and angle. To solve this problem, this paper proposes a method for facial expression recognition at different distances and angles, which solved the larger distance and deflection angle of facial expression recognition accuracy and real-time issues.

5.	**Raspberry Pi Based Emotion Recognition using OpenCV, TensorFlow, and Keras**

Reference: JOYDIP DUTTA. https://circuitdigest.com/microcontroller-projects/raspberry-pi-based-emotion-recognition-using-opencv-tensorflow-and-keras.

In this tutorial, they implement an Emotion Recognition System or a Facial Expression Recognition System on a Raspberry Pi 4. They apply a pre-trained model in order to recognize the facial expression of a person from a real-time video stream. The “FER2013” dataset is used to train the model with the help of a VGG-like Convolutional Neural Network (CNN).

6.	**Connect a Raspberry Pi or other device with AWS**

Reference: AWS. https://docs.aws.amazon.com/iot/latest/developerguide/connecting-to-existing-device.html.

This step-by-step tutorial guides you through all the steps you need to take in order to connect a Raspberry Pi or any other device with AWS. It explains to you how to set up the device, install the required tools and libraries for the AWS IoT Device SDK, install AWS IoT Device SDK, install and run the sample app, as well as view the messages from the sample app in the AWS IoT console.

7.	**Realtime Facial Emotion Recognition**

Reference: victor369basu. https://github.com/victor369basu/facial-emotion-recognition.

This repository demonstrates an end-to-end pipeline for real-time Facial emotion recognition application through full-stack development. The front end is developed in react.js and the back end is developed in FastAPI. The emotion prediction model is built with Tensorflow Keras, and for real-time face detection with animation on the front-end, Tensorflow.js has been used.

8.	**Kaggle FER-2013 DataSet**

Reference: MANAS SAMBARE. https://www.kaggle.com/datasets/msambare/fer2013.

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.
The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

9.	**Facial landmarks with dlib, OpenCV, and Python**

Reference: Adrian Rosebrock. https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/.

This post explains line by line the source code provided and demonstrates in detail what are facial landmarks and how to detect them using dlib, OpenCV, and Python. Also, it introduces alternative facial landmark detectors such as ones coming from the MediaPipe library which is capable of computing a 3D face mesh.

## V. Conclusion

Our project aimed to address the critical issue of stroke detection by developing an AI-powered system using facial expression analysis. Motivated by the importance of proactive healthcare and the time-sensitive nature of stroke intervention, we explored various machine learning algorithms, including Random Forest, SVM, and Vision Transformer, leveraging a dataset from Kaggle. While each algorithm had its merits and challenges, we ultimately found Amazon Rekognition to be a more efficient and accurate solution. Through this work, we envision a future where households actively safeguard health using integrated AI technologies, emphasizing the significance of early disease detection and prevention.
