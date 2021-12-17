import os
import glob

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument's to be adjusted for app
model_location='nyu.h5'
input_img_path='examples/*.png' # provide input filename or folder
input_img_path='examples/cj.png'

from PIL import Image
img = Image.open(input_img_path)
img = img.resize((240, 240), Image.ANTIALIAS)
print(img.size)
img.save("examples/cj.png")

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(model_location, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(model_location))

# Input images
#inputs = load_images( glob.glob(input_img_path) ) # all images of folder are gathered 
list_of_images=[input_img_path]
inputs = load_images(list_of_images) # all images of folder are gathered 
print("input images :",inputs.shape)

print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
print("output shape from model : ",outputs.shape)
#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   


plt.imshow(outputs[0],cmap="gray")
plt.show()

# viz = display_images(outputs.copy(), inputs.copy(), is_colormap=False )
# plt.figure(figsize=(10,5))
# plt.imshow(viz)
# plt.savefig('test.png')
# plt.show()

# def checkregion(x,y,w,h,depth_map):
#     img_size,

