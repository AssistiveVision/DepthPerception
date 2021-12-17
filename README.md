# Depth Perception 

# What is depth perception ? 
- Depth perception in simple words means to generate depth information from a given image (or video frame)
- Technically, it is a regression problem where we need to convert input pixels into appropiate depth value 

# How we approach this problem ?
- We have used the concept of auto-encoders to create depth maps of the input image frame
- We have used the base model architecture of DenseNet 169 which was pretrained on imagenet dataset.
- The Densenet 169 is used to downsample(encoding) the input image.
- Then we have defined our own architecture for upsampling (decoding) the encoded image

# Architecture For Decoder 
- We have used the following following order of layers and repeated it 4 times.
    - Billinear Upsampling 2D
    - Concatenate 
    - Conv2D
    - Leaky Relu
    - Conv2D
    - Leaky Relu 

# Loss Function Usage 
- Notmally loss function used for depth regression problems considers difference between ground truth depth map and predicted depth map values.
- However, we modified the loss function to incorporate issues caused by distortions caused by high frequncy details
- Our loss function estabilishes a balance between :-
    - Reconstructing depth images by minimizing the difference of depth
    - penalizing distorions of high frequency details in the image domain of depth map
- The 3 components of our loss function are -
    1. L1 Loss Function 
    2. L1 Image Gradient Based Loss Function 
    3. SSIM Loss Function 

# Challenges In Training 
- Since during training our main work was to reduce loss and loss values were quite high because the ground truth deoth values are bigger.
- To overcome this issue we used reciprocal of depth value as follows -:
    - Original Depth Map = y_original 
    - Target Depth Map = m/y_original 
        where m is the maximum depth in the scene ( For NYU dataset m=10)

# Resources
- Datset Link : https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- Basics Of Autoencoders : https://blog.keras.io/building-autoencoders-in-keras.html

# Model Architecture 
![(depth-perception-model-arch.png](Additional_Files/depth-perception-model-arch.png)

