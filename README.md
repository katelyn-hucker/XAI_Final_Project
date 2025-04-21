# XAI_Final_Project: Analyzing YOLOv8 performance on occluded cones and visible cones with explainable techniques.
## This project assesses LIME, SHAP, Class Activation Mapping (CAM) for the Nuscenes dataset with YOLOv8
## Dr. Brinnae Bent Duke AIPI 590
## Katie Hucker (kh509)

### Introduction:
The three methods are compared with 2 images. These 2 images were labeled in the Nuscenes dataset as either 1-40% visible cones or 80-100% visible cones. We test the modes on a fine-tuned YOLOv8 model which learned what cones and only cones were for a MIDS Capstone project. The visibilty levels were manually labeled by professional annotators. We wanted to assess what parts of the cone are most helpful or if YOLO shows bias between the occluded cones or visible cones. In addition, how do the three explainable methods do with the different visibility levels and with YOLO. 

### Navigating this Repo: 
There are 3 Google Colab Python Notebooks and the images I used to generate the results in this repo. Due to the output being so large Github had problems with the output being visible, so the notebooks just show the code. **However** If you click the "Open in Colab' widget it will take you to the notebooks where you can see the results! 

### Methodology and Runtimes
I ran the three explainable models. CAM was a very simple implmentation with few tweaks, however, this was done from scratch and without python packages. SHAP was ran at different batch size and sample counts to find an optimal number in terms of results and time. I found a nice middle ground there. LIME was ran similarly where different segmentation methods were tried as well as number of samples. SHAP was able to have a lot higher number of samples than LIME without crashing. CAM was the quickest, then SHAP, then LIME. Of course this can change with different settings and GPU availability. 

### Results
See my discussion for each explainable model within the notebooks. However across all three I found SHAP the most beneficial. It creates the superpixels which benefits this analysis how we can tell which parts of the cones were more helpful. It also provides a more helpful numeric output. SHAP also had minimal localized responses to areas which were NOT cones. Specifically around the cones when the cone ended SHAP had minimal response there -- a good sign. It also shows pretty good response when YOLO did not detect at all. 

### Reflection and What I Learned

I originally wanted to do a gradient vanilla saliency method however I learned YOLO v8 is not compatible with GRAD CAM and if it is only worked with baseline YOLO or the classification YOLO, not my finetuned version. In addition YOLOv8 was the first verison to have decentrealized anchors, where the box is drawn from the center so Anchors cannot work with it either! However, I contiued with the format doing a simpler CAM (which still showed great results!) and implementing SHAP which turned out to be my personal favorite. SHAP was user friendly and had understandable setting and output where the other ones either were more computationally heavy or had minimal interpretable out (looking at you CAM). I thought occlusion or visibility would play a bigger role, but it did not the cones showed response on all three methods when it was occluded or not. The explainable methods still found the cones when YOLO did not 'detect' so this means our YOLO detection confidence could be lowered as it might have found more cones accurately. I thought it was especially interesting to see a method hone in the shape of the cone or finding the bottom of the cone. These methods can be used to find what sorts of data we should be collecting or making sure it is correctly detection what it should be in the image. 

