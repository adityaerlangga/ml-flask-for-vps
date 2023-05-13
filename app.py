from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model1_name = "MM16"
model2_name = "VGG16"
model3_name = "ALEXNET"
model4_name = "LeNet"
model1 = load_model("Input/MM16_TA.h5")
model2 = load_model("Input/VGG16_TA.h5")
model3 = load_model("Input/1_Alexnet_TA32.h5")
model4 = load_model("Input/lenet.h5")

model1.make_predict_function()
model2.make_predict_function()
model3.make_predict_function()
model4.make_predict_function()

def predict_result(model_prediction, img_path):
	file_name = img_path.split("/")[-1].lower()
	predicted_class = np.argmax(model_prediction)
	probs = model_prediction[0]
	probabilitas = list(map(lambda x: round(x * 100), probs))
	if predicted_class == 0 and file_name.find("covid") != -1:
		result = "COVID-19"
		label = "BENAR"
	elif predicted_class == 0 and file_name.find("covid") == -1:
		result = "COVID-19"
		label = "SALAH"
	elif predicted_class == 1 and file_name.find("normal") != -1:
		result = "Normal"
		label = "BENAR"
	elif predicted_class == 1 and file_name.find("normal") == -1:
		result = "Normal"
		label = "SALAH"
	elif predicted_class == 2 and file_name.find("pneumonia") != -1:
		result = "Pneumonia"
		label = "BENAR"
	elif predicted_class == 2 and file_name.find("pneumonia") == -1:
		result = "Pneumonia"
		label = "SALAH"
	elif predicted_class == 3 and file_name.find("tbc") != -1:
		result = "TBC"
		label = "BENAR"
	else:
		result = "TBC"
		label = "SALAH"

	return result, label, probabilitas

def predict_label(img_path):
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img = tf.keras.preprocessing.image.img_to_array(img)/255
	img = np.expand_dims(img, axis=0)
	
	prediction1 = model1.predict(img)
	prediction2 = model2.predict(img)
	prediction3 = model3.predict(img)
	prediction4 = model4.predict(img)
	result1, label1, probabilitas1 = predict_result(prediction1, img_path)
	result2, label2, probabilitas2 = predict_result(prediction2, img_path)
	result3, label3, probabilitas3 = predict_result(prediction3, img_path)
	result4, label4, probabilitas4 = predict_result(prediction4, img_path)
	
	return result1, result2, result3, result4, label1, label2, label3, label4, probabilitas1, probabilitas2, probabilitas3, probabilitas4


# routes
@app.route("/", methods=['GET'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/Input/" + img.filename
		img.save(img_path)

		p1, p2, p3, p4, l1, l2, l3, l4, probs1, probs2, probs3, probs4 = predict_label(img_path)

	return render_template("index.html", filename = img.filename, prediction1 = p1,  prediction2 = p2, prediction3 = p3, prediction4 = p4,label1 = l1, label2 = l2, label3 = l3, label4=l4, img_path = img_path, model1_name = model1_name, model2_name = model2_name, model3_name = model3_name, model4_name = model4_name, probabilitas1 = probs1, probabilitas2 = probs2, probabilitas3 = probs3, probabilitas4 = probs4)


if __name__ =='__main__':
	app.run(host="0.0.0.0",debug=True)