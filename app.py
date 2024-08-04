from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import random

app = Flask(__name__)

conditions = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

# Load pre-trained VGG16 model
model_path = './model/best_model5.h5'
model = tf.keras.models.load_model(model_path)

target_size = (224,224)

def check_file_format(file):
    allowed_extensions = ['jpg', 'png', 'jpeg']

    file_format = '.' in file.filename and file.filename.rsplit('.', 1)[1].lower()

    upload_file_path = f"./static/upload/upload.{file_format}"

    file.save(upload_file_path)

    return  file_format in allowed_extensions, upload_file_path

def custom_preprocess_image(img):
    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Handle both RGB and grayscale images
    if len(img_array.shape) == 3:
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray_image = np.repeat(np.expand_dims(gray_image, axis=-1), 1, axis=-1)
    else:
        gray_image = img_array
        gray_image = np.repeat(np.expand_dims(img, axis=-1), 1, axis=-1)

    # Resize and normalize pixel values
    resized_image = cv2.resize(gray_image, target_size)
    normalized_image = resized_image / 255.0

    # Add batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image

def shuffle_images(predicted_condition = None):
    imgs = []

    file_path = os.path.dirname(__file__)
    base_path = os.path.join(file_path, "static", "dataset", "train")
    condition = predicted_condition if predicted_condition else random.choice(conditions)

    dataset_folder_path = os.path.join(base_path, condition)

    list_imgs = os.listdir(dataset_folder_path)

    if list_imgs:
        imgs.append(random.choice(list_imgs))
        imgs.append(random.choice(list_imgs))
        imgs.append(random.choice(list_imgs))
        imgs.append(random.choice(list_imgs))

        for idx, value in enumerate(imgs, 0):
            imgs[idx] = f"static/dataset/train/{condition}/{value}"

    return imgs

@app.route('/')
def index():
    imgs = shuffle_images()
    return render_template('classify.html', imgs=imgs)

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image file from the POST request
        image_file = request.files['image']

        allowed_file, upload_file_path = check_file_format(image_file)
        
        if not allowed_file:
            imgs = shuffle_images()
            return render_template('classify.html', validation_error=True, imgs=imgs)
        
        img = Image.open(image_file)

        # Preprocess the image
        img = img.resize(target_size)
        img_array = custom_preprocess_image(img)

        # Make predictions
        predictions = model.predict(img_array)
        
        predictions = np.array(predictions)
        predictions_labels = np.argmax(predictions)
        predictions_labels = conditions[predictions_labels]

        imgs = shuffle_images(predictions_labels)

        return render_template("classify.html", result=predictions_labels, imgs =imgs, upload_file_path= upload_file_path)

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
    base_path = os.path.join(file_path, 'static', 'dataset', dataset_type)

    for condition in conditions:
        path = os.path.join(base_path, condition)
        base_img_src = f"static/dataset/{dataset_type}/{condition}/"
        

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

@app.route("/accuracy")
def load_accuracy():
    return render_template("accuracy.html")

if __name__ == '__main__':
    app.run(debug=True, port=8000)
