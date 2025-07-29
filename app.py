import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fpdf import FPDF

# App & config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# Load model
model = load_model('skin_cancer_model.h5')

# Disease class labels
class_labels = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesions'
]

# Descriptions
disease_descriptions = {
    'Actinic Keratoses': 'Precancerous rough, scaly patches from sun exposure. Can develop into skin cancer.',
    'Basal Cell Carcinoma': 'A common form of skin cancer. Often appears as a slightly transparent bump.',
    'Benign Keratosis': 'Non-cancerous skin growths caused by aging or sun exposure.',
    'Dermatofibroma': 'A small, firm, often harmless skin nodule typically found on the legs or arms.',
    'Melanoma': 'Most serious type of skin cancer. Early detection is crucial.',
    'Melanocytic Nevi': 'Common moles or birthmarks. Usually harmless but should be monitored.',
    'Vascular Lesions': 'Blood vessel growths like hemangiomas or port-wine stains.'
}

# Home
@app.route('/')
def index():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions)
        predicted_class = class_labels[pred_idx]
        confidence = float(predictions[0][pred_idx])

        # PDF Report
        report_name = f"report_{uuid.uuid4().hex}.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_name)
        generate_pdf_report(report_path, predicted_class, confidence, disease_descriptions[predicted_class])

        return render_template(
            'result.html',
            predicted_class=predicted_class,
            confidence=round(confidence * 100, 2),
            description=disease_descriptions[predicted_class],
            filename=unique_filename,
            report_filename=report_name
        )

# Download report
@app.route('/download/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

# PDF Generation
def generate_pdf_report(path, disease, confidence, description):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Skin Disease Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Disease Predicted: {disease}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {round(confidence * 100, 2)}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Description:\n{description}")
    pdf.output(path)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
