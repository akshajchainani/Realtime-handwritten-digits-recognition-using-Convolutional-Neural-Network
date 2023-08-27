# Realtime Handwritten Digits Recognition using Convolutional Neural Network

![Uploading Realtime Digit Recognition - Google Chrome 2023-08-23 18-26-52.gif…]()


This repository contains the code and resources for a real-time handwritten digits recognition system using a Convolutional Neural Network (CNN). The system is built to recognize handwritten digits in real-time and display the predicted digit on the user interface.

## Table of Contents

- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model](#model)
- [Server](#server)
- [Training](#training)

## Introduction

Handwritten digits recognition is a fundamental problem in the field of machine learning and computer vision. This project demonstrates the use of a CNN to recognize handwritten digits in real-time. The user interface provides a canvas where users can draw digits, and the model predicts the digit based on the drawing.

## Folder Structure

- `deep_learning`: Contains Jupyter notebooks for implementing the Convolutional Neural Network, data preprocessing, and model evaluation.
  - `digit_deep_learning.ipynb`: A Jupyter notebook containing code and explanations for building, training, and evaluating the CNN model for handwritten digit recognition.
- `model`: This folder contains the pre-trained CNN model in TensorFlow.js format, including the `model.json` architecture file and two binary weight files.
- `server`: Includes the code for the server that handles real-time digit recognition requests.
  - `index.html`: An HTML file that provides a basic interface for drawing digits and getting real-time predictions.
  - `mnist_server.py`: A Python script that sets up a simple server to serve the HTML interface and handle digit recognition requests.
- `training`: Contains scripts and notebooks for training the CNN model on the MNIST dataset.
- `tfjs.html`: A simple HTML file that loads the pre-trained model using TensorFlow.js and provides the user interface for drawing and digit recognition.

## Setup

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Follow the instructions in the [Training](#training) section to train the CNN model or use the provided pre-trained model weights.

## Usage

1. Run the server by navigating to the `server` directory and executing `python mnist_server.py`.
2. Open the `index.html` file in a web browser.
3. Use the canvas on the web page to draw a handwritten digit.
4. The server will communicate with the CNN model to predict the drawn digit and display it on the UI.

## Model

The Convolutional Neural Network (CNN) architecture used for this project is designed to effectively recognize handwritten digits. The pre-trained model, available in the `model` folder, consists of the `model.json` architecture file and two binary weight files.

## Training

The `training` folder contains scripts and notebooks to train the CNN model on the MNIST dataset. You can follow the provided instructions in the `digit_deep_learning.ipynb` notebook to train the model from scratch or adapt the training process to your own dataset.

Feel free to use, modify, and distribute the code as per the terms of the license.
