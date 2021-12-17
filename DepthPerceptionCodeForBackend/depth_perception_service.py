import os

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from services.depth_perception.layers import BilinearUpSampling2D
from services.depth_perception.utils import predict, load_images, display_images

# Argument's to be adjusted for app
model_location='services/depth_perception/nyu.h5'



# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(model_location, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(model_location))

def predict_api(image_data):
    
    # Input images 
    #list_of_images=[input_img_path]
    #inputs = load_images(list_of_images) # all images of folder are gathered 
    inputs=image_data
    # Compute results
    outputs = predict(model, inputs)
    return outputs 


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # input_img_path='sample_img/img_1.png'
    # result=predict_api(input_img_path)
    # plt.imshow(result[0],cmap="gray")
    # plt.show()
    print("Ok")