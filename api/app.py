from image_detection import image_detection
import os
import flask
from flask import Flask, render_template, redirect, request, flash, url_for, jsonify
from werkzeug.utils import secure_filename
app = flask.Flask(__name__)
UPLOAD_FOLDER = '/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '290d1c47c5a94841bf35023c7ef8b7c7'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image_detection', methods=["GET", "POST"])
def detection():
    if request.method == "GET":
        return render_template("upload.html")
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"message":"No file part"})
        image = request.files["image"]
        if image.filename == '':
            return jsonify({"message":"No Selected Image"})
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save("uploads/"+filename)
            result = image_detection("./uploads")
            return jsonify({"data":result})
if __name__ == '__main__':
    app.run(debug=True)
