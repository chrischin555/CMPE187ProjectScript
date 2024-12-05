import os
import csv
from PIL import Image
import google.generativeai as genai
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# Setup for the API key and model
genai.configure(api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
model = genai.GenerativeModel('gemini-1.5-flash')

# Directories and filenames
IMAGES_FOLDER = 'Post_Impressionist_Painting_images'
VALID_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif')
CSV_INPUT_FILE = 'Post_Impressionist_Painting_input_expected.csv'
CSV_OUTPUT_FILE = 'Post_Impressionist_Painting_images_output.csv'

def is_similar(expected, generated, threshold=0.4):
    """Function to check similarity between expected and generated output"""
    ratio = SequenceMatcher(None, expected.lower(), generated.lower()).ratio()
    return ratio >= threshold

def process_images_from_csv():
    # Initialize counters for pass and fail
    pass_count = 0
    fail_count = 0

    # Read CSV file
    with open(CSV_INPUT_FILE, 'r') as csvfile:
        reader = list(csv.DictReader(csvfile))
        total_rows = len(reader)

        # Results dictionary; parse each row
        results = []
        for i, row in enumerate(reader, start=1):
            print(f"Processing row {i} of {total_rows}...")

            image_name = row['Image Name']
            input = row['Input']
            expected_output = row['Expected Output']
            image_path = os.path.join(IMAGES_FOLDER, image_name)

            # Case 1: If no image and no input, skip
            if image_name == "" and input == "":
                print(f"Skipping row {i} because both image and input are missing.")
                continue

            # Case 2: If no image, process only the input text
            if image_name == "" or not os.path.exists(image_path):
                print(f"No image or image not found for {image_name}. Processing text input...")

                # Generate response based on input text
                prompt = f"{input}."
                response = model.generate_content([prompt])
                generated_output = response.text.strip()

                passOrFail = 'Pass' if is_similar(expected_output, generated_output) else 'Fail'

                # Update pass/fail counts
                if passOrFail == 'Pass':
                    pass_count += 1
                else:
                    fail_count += 1

                # Record the result
                results.append({
                    'Image Name': image_name,
                    'Input': input,
                    'Expected Output': expected_output,
                    'Generated Output': generated_output,
                    'Pass/Fail': passOrFail
                })
                continue

            # Case 3: If image exists but no input, process the image alone
            if input == "":
                print(f"No input for image {image_name}. Processing image alone...")

                try:
                    # Open the image
                    test_image = Image.open(image_path)

                    # Generate response based on the image
                    prompt = f"Describe the image: {image_name}."
                    response = model.generate_content([prompt, test_image])
                    generated_output = response.text.strip()

                    passOrFail = 'Pass' if is_similar(expected_output, generated_output) else 'Fail'

                    # Update pass/fail counts
                    if passOrFail == 'Pass':
                        pass_count += 1
                    else:
                        fail_count += 1

                    # Record the result
                    results.append({
                        'Image Name': image_name,
                        'Input': input,
                        'Expected Output': expected_output,
                        'Generated Output': generated_output,
                        'Pass/Fail': passOrFail
                    })

                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
                    results.append({
                        'Image Name': image_name,
                        'Input': input,
                        'Expected Output': expected_output,
                        'Generated Output': f"Error: {e}",
                        'Pass/Fail': 'Fail'
                    })
                    fail_count += 1  # Increment fail count for errors

            else:
                # Case 4: Image and input exist, process both
                try:
                    # Open the image
                    test_image = Image.open(image_path)

                    # Generate response based on image and text
                    prompt = f"{input} The image is {image_name}."
                    response = model.generate_content([prompt, test_image])
                    generated_output = response.text.strip()

                    # Compare expected output with the generated output
                    passOrFail = 'Pass' if is_similar(expected_output, generated_output) else 'Fail'

                    # Update pass/fail counts
                    if passOrFail == 'Pass':
                        pass_count += 1
                    else:
                        fail_count += 1

                    # Record the result
                    results.append({
                        'Image Name': image_name,
                        'Input': input,
                        'Expected Output': expected_output,
                        'Generated Output': generated_output,
                        'Pass/Fail': passOrFail
                    })
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
                    results.append({
                        'Image Name': image_name,
                        'Input': input,
                        'Expected Output': expected_output,
                        'Generated Output': f"Error: {e}",
                        'Pass/Fail': 'Fail'
                    })
                    fail_count += 1  # Increment fail count for errors

    # Write the results to a new CSV file
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Input', 'Expected Output', 'Generated Output', 'Pass/Fail']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results have been saved to {CSV_OUTPUT_FILE}")
    print(f"Pass: {pass_count}, Fail: {fail_count}")

    # Plot the pass/fail rates
    plot_pass_fail_rate(pass_count, fail_count)

def plot_pass_fail_rate(pass_count, fail_count):
    # Generate a plot to visualize pass/fail rate
    labels = ['Pass', 'Fail']
    counts = [pass_count, fail_count]

    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Pass/Fail')
    plt.ylabel('Count')
    plt.title('Pass/Fail Rate')
    plt.show()

# Call the function to process images and generate plot
process_images_from_csv()
