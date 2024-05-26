# object-detection-project
Welcome to my GitHub repository for custom object detection using YOLOv8 by Ultralytics!

This project covers a range of object detection tasks and techniques, including utilizing a pre-trained YOLOv8-based network model for object detection, training a custom YOLOv8 model to recognize a single class (in this case, bikes, cars, person, truck).

To make this project accessible to all, I have leveraged Google Colab and Roboflow, providing easy-to-follow code and instructions for each stage of the project. Additionally, I have integrated my developed module test_image.py and test_vedio.py in folder object for object detection, tracking, and counting with YOLOv8, streamlining the object detection process for various applications.

Steps:
> Implement the given files in colab with GPU as run time
                  final_fog.ipynb, final_rain.ipynb, final_sand.ipynb,final_snow.ipynb
> Download the best weights from the obtained results.
> Use the best weights to detect objects
> object detection code is in folder /object/
The best weight obtained from above training models are capture in folder object





names:
  0: Bike
  1: car
  2: person
  3: truck
