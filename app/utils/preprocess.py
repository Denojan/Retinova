from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)  

def preprocess_image(file):

    img = Image.open(file.stream).convert("RGB")
    img = img.resize(IMG_SIZE) 
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array
