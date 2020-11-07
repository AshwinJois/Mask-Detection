# Mask-Detection-Using-CNN-and-OpenCV
*This CNN tells us if a person is wearing  a mask or not!*


Hello,

We all know how Deep learning is dominating in the field of Computer Vision!! It is a technique that teaches network to do a specific task. The reason behind so many innovations in these days are because of Deep Learning. Some examples are  driverless cars, voice control on our phones, tablets and TVs. Deep Learning is getting lots of attention and is achieving many tasks which could not be achieved before. 
Here, I have explained how to detect a Person with and without mask in an image using a Convolutional Neural Network. To make the detections even more attractive I have also used bounding boxes. 

**Understanding Convolutional Neural Network:**

A CNN is a Neural Network used to extract some features of the Image. The CNN is capable of differentiating the images given to it and figure out the unique features in them. CNN takes the Image’s raw pixel data, trains the model and then extracts the features for classification. 

![1_cot55wd6gdoJlovlCw0AAQ](https://user-images.githubusercontent.com/63425115/98447050-3ebb5080-2122-11eb-8ebf-15d1bd8f8958.png)

When we look at a picture of an Elephant, we can classify it by looking at the identifiable features such as ears, face , trunk, tusks or 4 legs. In a similar way, the computer is able perform image classification by looking for low level features such as edges and curves, and then building up to more abstract concepts through a series of convolutional layers. This is a general overview of what a CNN does. 



**Architecture of a CNN:**

A CNN is composed of several kinds of layers: Convolutional Layer, Pooling Layer, Fully connected input layer, Fully connected layer, Fully Connected output layer. Check https://missinglink.ai/guides/convolutional-neural-networks/convolutional-neural-network-architecture-forging-pathways-future/#:~:text=CNN%20architecture%20is%20inspired%20by,or%20feature%20of%20the%20image.  to know in detail.

![LeNet-5-1998](https://user-images.githubusercontent.com/63425115/98447108-95c12580-2122-11eb-8772-342bda44b4d6.png)

**Dataset:**

Whenever we train a Neural Network it is always good to have a suffiecient amount of Dataset. This link https://github.com/prajnasb/observations/tree/master/experiements/data will help you download Images with and without mask.
While training a Neural Network the dataset is split into Training set and Validation or Test set. Usually 80% of the entire dataset goes to Training set and the remaining 20% goes to Validation set. Check https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7 for more information.


**Face Detection using OpenCV:**

OpenCV already contains many pre-trained classifier for face, eyes etc. It comes with a trainer as well as a detector. Here is the link to OpenCV face detection using Haar Cascades https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html. Make sure you download the .xml file.


**Result Analysis:**

The model was trained for 10 and 15 epochs, the bheavior of Accuracy and Validation Accuracy is as shown in the graph below.

![abc](https://user-images.githubusercontent.com/63425115/98447193-2861c480-2123-11eb-9dd5-61599a48edb5.JPG)

(The model was able to predict properly for 10 epochs itself)


**Output:** 

After training I have predicted on few images and have put all of them together. (Chuck the bluish color on the output, the reason behind it is OpenCV uses BGR as its default colour order for images, matplotlib uses RGB. When you display an image loaded with OpenCv in matplotlib the channels will be back to front)

![IMG_20201107_164410](https://user-images.githubusercontent.com/63425115/98447233-6eb72380-2123-11eb-8a95-48590c158535.jpg)


**Thanks!!**




