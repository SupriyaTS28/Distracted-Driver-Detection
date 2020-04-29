# Distracted-Driver-Detection

# Overview:
Distracted driving is defined as any activity that takes away your attention from driving, whether it be talking on a cell phone, texting, operating a radio, eating, or talking to a passenger. In 2018, there were 2841 deaths and around 400,000 people in the United States injured as a result of crashes involving distracted drivers. As there are several different forms of distracted driving, it is important to be able to recognize and correctly classify each of them in order to minimize car accidents.

We explore different deep learning techniques in order to build a high accuracy model that detects and correctly distinguishes between several distracted driving activities such as texting, talking on the phone, operating the radio, drinking, reaching behind, adjusting hair and makeup, and talking to a passenger.

Dataset: State Farm Distracted Driver Detection (available on Kaggle- https://www.kaggle.com/c/state-farm-distracted-driver-detection)
Training set ~22.4 K and 79.7 K unlabeled test samples. This is a publicly available dataset from State-Farm. It consists of images of driver doing 1 of 10 actions listed below.

10 classes(driver actions):
1.	safe driving
2.	texting - right
3.	talking on the phone - right
4.	texting - left
5.	talking on the phone - left
6.	operating the radio
7.	drinking
8.	reaching behind
9.	hair and makeup
10.	talking to passenger

Methods:
In this project we compare three approaches to solve an Image classification problem with small labeled dataset.

The three proposed methods are:
1. Data Augmentation: Train a model from scratch with augmented data
2. Transfer Learning:  Use a pre-trained model and fine tune it.
3. Semi-supervised Learning: Use semi-supervised approach to utilize the 79.7 K unlabeled samples.

# Team
Bhagya Shree Kottoori, Rosa Ku, Supriya Tumkur Suresh Kumar
