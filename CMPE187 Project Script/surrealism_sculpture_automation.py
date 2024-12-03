# TODO: keep count of pass and fail, then generate a plot with the pass/fail rate
import os
import csv
from PIL import Image
import google.generativeai as genai

genai.configure(api_key ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

IMAGES_FOLDER = 'surrealism_sculpture_images'
VALID_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif')
CSV_INPUT_FILE = 'surrealism_sculpture_input_expected.csv'
CSV_OUTPUT_FILE = 'surrealism_sculpture_output.csv'

# test_image = PIL.Image.open('surrealism_sculpture_images\\lobster telephone dark.jpeg')
model = genai.GenerativeModel('gemini-1.5-flash')
# response = model.generate_content(["Tell me who made this sculpture.", test_image])
# print(response.text)

def process_images_from_csv():
    # read CSV file
    with open(CSV_INPUT_FILE, 'r') as csvfile:
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
            image_path = os.path.join(IMAGES_FOLDER, image_name)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image {image_name} not found...")

                # Append each row to results dictionary
                results.append({
                    'Image Name': image_name,
                    'Input': input,
                    'Expected Output': expected_output,
                    'Actual Output': 'Image not found',
                    'Pass/Fail': 'Fail'
                })
                continue

            # Try/Catch block for generating response for each image
            try:
                # Open the image
                test_image = Image.open(image_path)

                # Generate response based on image
                prompt = f"{input} The image is {image_name}."
                response = model.generate_content([prompt, test_image])
                generated_output = response.text.strip()

                passOrFail = 'Pass' if expected_output.lower() in generated_output.lower() else 'Fail'

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
                    'Match': 'No'
                })
    # Write the results to a new CSV file
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldNames = ['Image Name', 'Input', 'Expected Output', 'Generated Output', 'Pass/Fail']
        writer = csv.DictWriter(csvfile, fieldnames = fieldNames)

        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results have been saved to {CSV_OUTPUT_FILE}")

# for file_name in os.listdir(IMAGES_FOLDER):
#     if file_name.lower().endswith(VALID_EXTENSIONS):  # Check for valid image file extensions
#         file_path = os.path.join(IMAGES_FOLDER, file_name)
#         try:
#             # Open the image
#             test_image = Image.open(file_path)
            
#             # Generate content for the image
#             prompt = f"Tell me who made this sculpture in the image: {file_name}"
#             response = model.generate_content([prompt, test_image])
            
#             # Print the response
#             print(f"Response for {file_name}: {response.text}")
#         except Exception as e:
#             print(f"Failed to process {file_name}: {e}")

# Call function
process_images_from_csv()