import os

from ultralytics import YOLO

model = YOLO('best.pt')
print(os.getcwd())
os.chdir('../')
print(os.getcwd())        # change directory path as needed
detection_output = model.predict("./images", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())







