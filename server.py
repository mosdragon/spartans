import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename

from main import merge_images
import cv2

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def generate_merge():
    pass

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():

    if 'grouptag' in request.form:
        tag = request.form['grouptag']
    else:
        tag = 'default'

    tag = tag.strip()
    directory = app.config['UPLOAD_FOLDER'] + "tag_" + tag + "/"
    print "DIRECTORY: {}".format(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    merged_img_name = directory + ".jpg"

    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup

        file.save(os.path.join(directory, filename))
        
        # Now, allow it to combine
        output_filename = "tag_" + tag + ".jpg"
        if len(os.listdir(directory)) <= 1:
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], merged_img_name))
            img = cv2.imread(directory + filename)
        else:
            # Combine with other images
            img = merge_images(path=directory)
        
        dest_file = app.config['UPLOAD_FOLDER'] + output_filename
        print "DEST FILE: {}".format(dest_file)
        cv2.imwrite(dest_file, img)

        return redirect(url_for('uploaded_file',
                                filename=output_filename))

    return redirect(url_for('index'))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("8080"),
        debug=True
    )
