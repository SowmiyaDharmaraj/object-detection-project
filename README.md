# object-detection-project
Welcome to my GitHub repository for custom object detection using YOLOv8 by Ultralytics!

This project covers a range of object detection tasks and techniques, including utilizing a pre-trained YOLOv8-based network model for object detection, training a custom YOLOv8 model to recognize a single class (in this case, bikes, cars, person, truck).

To make this project accessible to all, I have leveraged Google Colab and Roboflow, providing easy-to-follow code and instructions for each stage of the project. Additionally, I have integrated my developed module test_image.py and test_vedio.py in folder object for object detection, tracking, and counting with YOLOv8, streamlining the object detection process for various applications.

The training process is automated for efficient training of custom object detection models. Simply specify detectable classes and training hyperparameters, and the code will take care of the rest, including downloading proper datasets, reorganizing the dataset in the YOLO-compatible format, generating proper YAML files, starting the training process, and automatically saving the results.

Navigating this repository
Custom-object-detection-with-YOLOv8: Directory for training and testing custom object detection models basd on YOLOv8 architecture, it contains the following folders files:

final_fog.ipynb, final_rain.ipynb, final_sand.ipynb,final_snow.ipynb: an implementation example for the trained models.

The best weight obtained from above training models are capture in folder 

Custom object detection using YOLOv8/Rain training results for epoch 100/results.csv
Data collection:
To collect diverse and representative data for object detection using YOLOv8, or generally any other object detection model, the Open Images library provides a valuable resource that includes millions of well-labeled images with a wide range of object classes.

For more details about how to download and understand data provided by this library chech the following link.
For the rest of this data collection section, all data will be downloaded programatically (in script, no need for manual download).
Downloading annotations and metadata for training, validation and (optional) testing
Let's first start by downloading training, validation and testing annotations and metadata.

# training annotations and metadata
!wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv # 2.1G

# validation annotations and metadata
!wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv # 21M

# testing annotations and metadata
!wget https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv # 71M
oidv6-train-annotations-bbox.csv , validation-annotations-bbox.csv and test-annotations-bbox.csv are csv files that have training, validation and test metadata. All these files follow the same format. For more details check the Open Images Dataset formats.

ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,XClick1X,XClick2X,XClick3X,XClick4X,XClick1Y,XClick2Y,XClick3Y,XClick4Y
P.S : At this point we have only downloaded the metadata CSV files and not the actual image files. The reason for this is that we only need a specific subset of the Open Images dataset for our target objects, and downloading the entire dataset of 1.9 million images would be both time-consuming and unnecessary. By downloading only the necessary metadata files and selecting a subset of the images, we can save time and storage space while still obtaining high-quality data for our YOLOv8 model.

Selecting a Sub-Dataset for Object Detection: Choosing the Right Data for Your YOLOv8 Model
For more dtails about this important part of data collection check the Open Images Download Section
This section will explain the main strategy behind building a sub-dataset, with image data, for specific objects we want our model to detect.
We will simply follow the Open Image guidelines. The main approach at this point is to create a text file, image_list_file.txt containing all the image IDs that we're interested in downloading. These IDs come from filtering the annotations with certain classes. The text file must follow the following format : $SPLIT/$IMAGE_ID, where $SPLIT is either "train", "test", "validation"; and $IMAGE_ID is the image ID that uniquely identifies the image. A sample file could be:

train/f9e0434389a1d4dd
train/1a007563ebc18664
test/ea8bfd4e765304db
Now let's get to the part where we actually download images. Open Images provided a Python script that downloads images indicated by the image_list_file.txt we just created. First we download downloader.py by executing the following command :


Training the model
Now that our dataset is ready let's get to the training part

Preparing the configuration YAML file
In order to train a YOLOv8 model for object detection, we need to provide specific configurations such as the dataset path, classes and training and validation sets. These configurations are typically stored in a YAML (Yet Another Markup Language) file which serves as a single source of truth for the model training process. This allows for easy modification and replication of the training process, as well as providing a convenient way to store and manage configuration settings.

This YAML file should follow this format:

path: /mydrive/working/data  # Use absolute path 
train: images/train
val: images/validation

names:
  0: Bike
  1: car
  2: person
  3: truck
