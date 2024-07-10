# importing  the libraries
import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# initializing the flask app with name
app = Flask(__name__)

# load model 
model = load_model("mushroom.h5")

#Defines a route for the root URL ("/") and associates it with the index() function.
#The function renders the "index.html" template
@app.route("/")
def index():
    return render_template("index.html")

#Defines a route for the root URL ("/") and associates it with the home() function.
#The function renders the "index.html" template
@app.route("/home")
def home():
    return render_template('index.html')

#Defines a route for the root URL ("/") and associates it with the input() function.
#The function renders the "input.html" template
@app.route("/input")
def input():
    return render_template('input.html')


# variable with empty value for storing the result
result = ' '


#Defines a route for the root URL ("/") and associates it with the predict() function.
#The function renders the "output.html" template for predicting the result
@app.route('/predict', methods=['GET','POST'])
def predict():
    global result
    if request.method == "POST":
        #exception handling code block starts here
        try:
            f = request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)
            prediction = np.argmax(model.predict(img_data), axis=1)
            index = ['Boletus', 'Lactarius', 'Russula']
            result = str(index[prediction[0]])
        except Exception as e:
            result = 'Error: ' + str(e)
    return render_template('output.html', result=result)


#Defines a route for the root URL ("/") and associates it with the images() function.
#The function renders the "images.html" template with images of mushroom
@app.route("/images")
def images():
    return render_template('images.html')


# making the server to debug and it's value is True
if __name__ == "__main__":
    app.run(debug=True)

