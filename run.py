import os
import sys
import subprocess
import time
import torch
import torch.onnx
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

model = torch.hub.load('pytorch/vision:v0.11.2', 'resnet50', pretrained=True) #pretrained=True)

model.eval()
print(model)

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print("input size : ", input_batch.size())

input_batch_np = input_batch.cpu().numpy()

torch_results = []

# using pytorch gpu
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.init()
    model.cuda()
    cuda_a = input_batch.cuda()
    cuda_out = model(cuda_a)

print(cuda_out.size())

torch.cuda.empty_cache()

model.to("cpu")

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
tf.keras.backend.set_image_data_format('channels_first')
# ok_model = ResNet50(include_top=False, input_shape=np.shape(input_batch_np)[1:])
ok_model = ResNet50(weights= 'imagenet')

img_path = './dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = ok_model.predict(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

converter = tf.lite.TFLiteConverter.from_keras_model(ok_model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open('tflite_model.tflite', 'wb').write(tflite_model)


interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, x)
interpreter.invoke()

tflite_out = interpreter.get_tensor(output_index)
print('Predicted:', decode_predictions(tflite_out, top=3)[0])

for i in range(20):
    print(tflite_out[0][i], " / ", preds[0][i])



