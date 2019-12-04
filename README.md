# Object-recognition-in-Videos-
Object recognition in Videos using Faster RCNN and SORT

## Objective
Video-based recognition of objects in people's hands 

## Dataset
1. Frames of videos captured in Folder 'images' 
2. A text file dictionary for every frame in Folder 'dicts'
Each file consisting of following

img/image_sample_name.txt
image_width image_height

label
bounding box - x y width height

For an image file img.jpg, a dictionary img.txt exists

## Running instructions

python main.py --classes "obj1,obj2" [OPTION] 

--epochs - Epochs for training
--batch_size - Batch size for training
--batch_size_test - Batch size for testing
--classes - String with class dictionary

