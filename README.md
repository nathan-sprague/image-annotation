# image-annotation

Two relatively simple annotation tools. At the top, of the file, set videosPath to the path of the videos and images that you want to annotate. There are a few other parameters you can change in the code if you'd like.

Run the code and click around to make the annotations. It will automatically save the annotations as json when you move to the next file or close the program.

You can point it to a YOLO model and it can make suggested annotations.

## makeAnnotationsBoundingBox
Use to make annotations for the bounding boxes. You can set the number of keypoints to annotate as well.

## makeAnnotationsSegmentation
Use to make annotations for segmentation models.