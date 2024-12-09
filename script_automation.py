import os
import csv
from PIL import Image
from difflib import SequenceMatcher
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import zipfile

# REMEMBER TO X OUT API KEY, THIS IS MY API KEY AND I DON'T WANT ANYTHING TO HAPPEN IF SOMEONE STEALS IT.
genai.configure(api_key ="AIzaSyCeihAQWc3hN_FgPwE0ALeX3WH5Oimk_8I")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# IMAGES_FOLDER = 'surrealism_sculpture_images'
# VALID_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif')
# CSV_INPUT_FILE = 'surrealism_sculpture_input_expected.csv'
# CSV_OUTPUT_FILE = 'surrealism_sculpture_output.csv'

# os.makedirs(IMAGES_FOLDER, exist_ok=True)

# test_image = PIL.Image.open('surrealism_sculpture_images\\lobster telephone dark.jpeg')
model = genai.GenerativeModel('gemini-1.5-flash')
# response = model.generate_content(["Tell me who made this sculpture.", test_image])
# print(response.text)


def extract_zip(zipPath, extractTo):
    """Extract the uploaded ZIP file."""
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(extractTo)

def is_similar(expected, generated, threshold=0.05):
    similarity_ratio = compute_cosine_similarity(expected, generated)
    return similarity_ratio >= threshold

def compute_cosine_similarity(expected, generated):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([expected.lower(), generated.lower()])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0, 1]

def results_chart(pass_fail_stats):
    """Display a pie chart of pass/fail results."""
    numPass = pass_fail_stats['numPass']
    numFail = pass_fail_stats['numFail']
    labels = ['Pass', 'Fail']
    sizes = [numPass, numFail]
    colors = ['green', 'red']

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Pass/Fail Results')

    os.makedirs('static', exist_ok=True)

    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()

def process_images_from_csv(csv_path, images_folder, output_path):
    numPass = 0
    numFail = 0
    folderStats = {}
    # read CSV file
    with open(csv_path, 'r') as csvfile:
        reader = list(csv.DictReader(csvfile))

        # to keep track of how many rows im currently at, remove later
        total_rows = len(reader)

        # Results dictionary; parse each row
        results = []
        for i, row in enumerate(reader, start=1):
            # Progress counter
            print(f"Processing row {i} of {total_rows}...")

            image_name = row['Image Name']
            input = row['Input']
            expected_output = row['Expected Output']
            # image_path = os.path.join(IMAGES_FOLDER, image_name)

            found = False
            for root, _, files in os.walk(images_folder):
                 if image_name in files:
                    image_path = os.path.join(root, image_name)
                    found = True
                    folder_name = os.path.basename(root)

                    # Open image and process
                    try:
                        test_image = Image.open(image_path)
                        prompt = f"{input} The image is {image_name}."
                        response = model.generate_content([prompt, test_image])
                        generated_output = response.text.strip()

                        passOrFail = 'Pass' if is_similar(expected_output, generated_output) else 'Fail'
                        if passOrFail == 'Pass':
                            numPass += 1
                        else:
                            numFail += 1

                        # Update folder stats
                        folderStats.setdefault(folder_name, {'Pass': 0, 'Fail': 0})
                        folderStats[folder_name][passOrFail] += 1

                        # Record the result
                        results.append({
                            'Image Name': image_name,
                            'Input': input,
                            'Expected Output': expected_output,
                            'Generated Output': generated_output,
                            'Pass/Fail': passOrFail,
                            'Folder': folder_name
                        })
                    except Exception as e:
                        results.append({
                            'Image Name': image_name,
                            'Input': input,
                            'Expected Output': expected_output,
                            'Generated Output': f"Error: {e}",
                            'Pass/Fail': 'Fail',
                            'Folder': folder_name
                        })
                    break

            if not found:
                results.append({
                    'Image Name': image_name,
                    'Input': input,
                    'Expected Output': expected_output,
                    'Generated Output': 'Image not found',
                    'Pass/Fail': 'Fail',
                    'Folder': 'N/A'
                })
                numFail += 1

    # Write results to CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Image Name', 'Input', 'Expected Output', 'Generated Output', 'Pass/Fail', 'Folder']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return {'numPass': numPass, 'numFail': numFail, 'folder_stats': folderStats}
        
@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        image_zip = request.files.get('image_zip')

        if not csv_file or not image_zip:
            return "Please upload both a CSV file and a ZIP file of images.", 400

        # Save uploaded files
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.csv')
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images.zip')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        csv_file.save(csv_path)
        image_zip.save(zip_path)

        # Extract ZIP file
        extract_to = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
        os.makedirs(extract_to, exist_ok=True)
        extract_zip(zip_path, extract_to)

        # Process files
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
        pass_fail_stats = process_images_from_csv(csv_path, extract_to, output_path)

        print("DEBUG:", pass_fail_stats)

        # Generate the chart
        results_chart(pass_fail_stats)

        # Display results
        return render_template('results.html', pass_fail_stats=pass_fail_stats, chart_filename = 'chart.png')

    return render_template('upload.html')

@app.route('/download')
def download_output():
    """Download the output CSV file."""
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
    return send_file(output_path, as_attachment=True)

# Call function
# process_images_from_csv()
# passFailDisplay()

if __name__ == '__main__':
    app.run(debug=True)