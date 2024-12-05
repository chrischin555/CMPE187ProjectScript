import os
import csv
from PIL import Image
import google.generativeai as genai
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup for the API key and model
genai.configure(api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Directories and filenames
IMAGES_FOLDER = 'Post_Impressionist_Painting_images'
VALID_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif')
CSV_INPUT_FILE = 'Post_Impressionist_Painting_input_expected.csv'
CSV_OUTPUT_FILE = 'Post_Impressionist_Painting_output.csv'

def compute_tfidf_similarity(expected, generated, threshold=0.3):
    """Compute similarity using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer().fit_transform([expected, generated])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    print(f"TF-IDF Cosine Similarity: {similarity}")
    return similarity >= threshold

def process_images_from_csv():
    """Main function to process images and inputs from the CSV file."""
    pass_count = 0
    fail_count = 0

    if not os.path.exists(CSV_INPUT_FILE):
        print(f"Error: Input CSV file '{CSV_INPUT_FILE}' does not exist.")
        return

    results = []

    with open(CSV_INPUT_FILE, 'r') as csvfile:
        reader = list(csv.DictReader(csvfile))
        total_rows = len(reader)

        for i, row in enumerate(reader, start=1):
            print(f"Processing row {i} of {total_rows}...")

            image_name = row.get('Image Name', "").strip()
            input_text = row.get('Input', "").strip()
            expected_output = row.get('Expected Output', "").strip()
            image_path = os.path.join(IMAGES_FOLDER, image_name) if image_name else ""

            # Log the constructed image path
            print(f"Constructed image path: {image_path}")

            if not image_name and not input_text:
                print(f"Row {i}: Skipping because both image and input are missing.")
                continue

            try:
                # Handle text-only inputs
                if not image_name or not os.path.exists(image_path):
                    print(f"Row {i}: No image or image not found for '{image_name}'. Processing text input...")
                    generated_output = process_text_only(input_text)
                
                # Handle image-only inputs
                elif not input_text:
                    print(f"Row {i}: No text input for image '{image_name}'. Processing image alone...")
                    generated_output = process_image_only(image_path, image_name)
                
                # Handle combined image and text inputs
                else:
                    print(f"Row {i}: Processing combined image and text for '{image_name}'...")
                    generated_output = process_image_and_text(image_path, image_name, input_text)

                # Compare expected and generated outputs
                passOrFail = 'Pass' if compute_tfidf_similarity(expected_output, generated_output) else 'Fail'
                if passOrFail == 'Pass':
                    pass_count += 1
                else:
                    fail_count += 1

                results.append({
                    'Image Name': image_name,
                    'Input': input_text,
                    'Expected Output': expected_output,
                    'Generated Output': generated_output,
                    'Pass/Fail': passOrFail
                })

            except Exception as e:
                print(f"Error processing row {i} ({image_name}): {e}")
                results.append({
                    'Image Name': image_name,
                    'Input': input_text,
                    'Expected Output': expected_output,
                    'Generated Output': f"Error: {e}",
                    'Pass/Fail': 'Fail'
                })
                fail_count += 1

    # Write results to the output CSV
    write_results_to_csv(results)
    print(f"Results have been saved to '{CSV_OUTPUT_FILE}'")
    print(f"Pass: {pass_count}, Fail: {fail_count}")
    plot_pass_fail_rate(pass_count, fail_count)

def process_text_only(input_text):
    """Generate output for text-only inputs."""
    prompt = f"{input_text}."
    response = model.generate_content([prompt])
    return response.text.strip()

def process_image_only(image_path, image_name):
    """Generate output for image-only inputs."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return "Error: Image not found"

    try:
        with Image.open(image_path) as img:
            img.verify()  # Validate the image
            print(f"Image is valid: {image_name}")

        with Image.open(image_path) as img:
            prompt = f"Describe the image: {image_name}."
            response = model.generate_content([prompt])
        return response.text.strip()
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return f"Error: {e}"

def process_image_and_text(image_path, image_name, input_text):
    """Generate output for combined image and text inputs."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return "Error: Image not found"

    try:
        with Image.open(image_path) as img:
            img.verify()  # Validate the image
            print(f"Image is valid: {image_name}")

        with Image.open(image_path) as img:
            prompt = f"{input_text} The image is {image_name}."
            response = model.generate_content([prompt])
        return response.text.strip()
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return f"Error: {e}"

def write_results_to_csv(results):
    """Write the results to the output CSV file."""
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Input', 'Expected Output', 'Generated Output', 'Pass/Fail']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def plot_pass_fail_rate(pass_count, fail_count):
    """Generate a plot to visualize the pass/fail rate."""
    labels = ['Pass', 'Fail']
    counts = [pass_count, fail_count]

    plt.bar(labels, counts, color=['green', 'red'])
    
    # Annotate the bars with their counts
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.xlabel('Pass/Fail')
    plt.ylabel('Count')
    plt.title('Pass/Fail Results')
    plt.show()

# Execute the main function
if __name__ == "__main__":
    process_images_from_csv()
