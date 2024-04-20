from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.applications.vgg16 import decode_predictions

app = Flask(__name__)

conditions = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

# Load pre-trained VGG16 model
model_path = './model/model224dataset550100epoch.h5'
model = tf.keras.models.load_model(model_path)

target_size = (224,224)
def custom_preprocess_image(img):
    # Convert the image to a NumPy array
    img_array = np.array(img)
    print("mulai preprocessing",img_array.shape)

    # Handle both RGB and grayscale images
    if len(img_array.shape) == 3:
        print("masuk if")
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        print("gray if", gray_image.shape)
        gray_image = np.repeat(np.expand_dims(gray_image, axis=-1), 3, axis=-1)
    else:
        gray_image = img_array
        gray_image = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)

    # Resize and normalize pixel values
    print("mulai resize")
    resized_image = cv2.resize(gray_image, target_size)
    print("selesai resize")
    normalized_image = resized_image / 255.0

    print("normal shape", normalized_image.shape)

    # Add batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    print("preproceseed shape", preprocessed_image.shape)

    return preprocessed_image


@app.route('/')
def index():
    return render_template('classify.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image file from the POST request
        image_file = request.files['image']
        img = Image.open(image_file)

        # Preprocess the image
        img = img.resize(target_size)
        img_array = custom_preprocess_image(img)

        # Make predictions
        predictions = model.predict(img_array)
        
        predictions = np.array(predictions)
        predictions_labels = np.argmax(predictions)
        predictions_labels = conditions[predictions_labels]


        return render_template("classify.html", result=predictions_labels)

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()

        print("\ntraceback", traceback_str)
        return jsonify({'error': str(e)})

@app.route("/dataset")
def load_dataset():
    dataset_type = request.args.get("data")

    list_html = """<div class="accordion" id="accordionExample">"""
    file_path = os.path.dirname(__file__)
    base_path = os.path.join(file_path, 'static', 'dataset2', dataset_type)

    for condition in conditions:
        path = os.path.join(base_path, condition)
        base_img_src = f"static/dataset2/{dataset_type}/{condition}/"
        

        list_html += f"""
        <div class="accordion-item" style="width:100%;">
            <h2 class="accordion-header">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{"collapse" + str(conditions.index(condition))}" aria-expanded="false" aria-controls="{"collapse" + str(conditions.index(condition))}">
                    {condition}
                </button>
            </h2>
            <div id="{"collapse"+ str(conditions.index(condition))}" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
                <div class="accordion-body">
                    <div class="row row-cols-auto">
        """        

        for file_name in os.listdir(path):
            img_src = base_img_src+file_name
            list_html+=f"""
            <div class="col">
                <img src="{img_src}" class="img-thumbnail" alt="{file_name}" style="width:150px; height:150px;">
            </div>
            """
        list_html+="</div></div></div>"

    list_html+="</div>"
    return render_template("dataset.html", list_html=list_html)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
