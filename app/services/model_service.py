from ..services.model.load_model import get_amd_model
from ..utils.preprocess import preprocess_image
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.nasnet import preprocess_input
# model = get_model()  
model_AMD = get_amd_model()  


# model = get_model()  

# def predict_image(file):
 
#     img_array = preprocess_image(file)

#     predictions = model.predict(img_array)

#     predicted_class = np.argmax(predictions[0])

#     print(predictions)
#     print(predicted_class)

#     class_labels = ["NPDR", "No_DR", "PDR"]

#     print(class_labels[predicted_class])

#     predicted_label = class_labels[predicted_class]

#     return {
#         "label": predicted_label,
#         "confidence": predictions[0].tolist()
#     }


def preprocess_image(file, target_size=(331, 331)):
    try:
        
        img = Image.open(file).convert("RGB")

        
        img = img.resize(target_size)

        
        img_array = np.array(img)

        
        img_array = preprocess_input(img_array)

        
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


# Prediction function
def predict_image_amd(file):
  
    try:
        # if not is_valid_image(file):
        #     return {"error": "Invalid image format. Supported formats: JPEG, PNG, BMP."}

        # Preprocess the image
        img_array = preprocess_image(file)

        # Make predictions
        predictions = model_AMD.predict(img_array)

        # Determine the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]

       
        class_labels = ["dry", "normal", "wet"]
        predicted_label = class_labels[predicted_class]
        # confidence = [round(float(conf), 4) for conf in predictions[0]]
        confidence = round(float(predictions[0][predicted_class]), 4)


        return {
            "disease":"Age related macular degenration",
            "label": predicted_label,
            # "confidence list": predictions[0].tolist(),
            "confidence": confidence,

        }
    except FileNotFoundError as e:
        return {"error": "File not found.", "details": str(e)}
    except tf.errors.InvalidArgumentError as e:
        return {"error": "Invalid model input.", "details": str(e)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}