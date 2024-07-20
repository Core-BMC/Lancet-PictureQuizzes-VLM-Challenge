import anthropic
import base64
import io
import os
import pandas as pd
import statistics
import sys
import time
from PIL import Image
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.getcwd())
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)
os.environ['ANTHROPIC_API_KEY'] = os.getenv("MY_ANTHROPIC_API_KEY")


class ClaudeVisionAnalyzer:
    def __init__(self, api_key, time_file_name="Claude_execution_times.xlsx"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.execution_times = []
        self.time_file_name = os.path.join('time', time_file_name)
        self.df_execution_times = self.load_or_initialize_execution_times()
        self.log_file_path = os.path.join("./", "process_log.txt")
        self.temperatures = [0]
        self.base_result_folder = "Claude_result/Claude_result"
        self.max_try = 1
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.case_folder = os.path.join(parent_dir, "Lancet_IMAGE240508")

    def load_or_initialize_execution_times(self):
        if os.path.exists(self.time_file_name):
            print("Loading time file")
            return pd.read_excel(self.time_file_name)
        else:
            print("Creating time file")
            return pd.DataFrame({
                'number': pd.Series(dtype='int'),
                'temperature': pd.Series(dtype='float'),
                'try': pd.Series(dtype='int'),
                'time': pd.Series(dtype='float')
            })

    def save_execution_times_to_excel(self):
        self.df_execution_times.to_excel(self.time_file_name, index=False)
        print(f"Execution times saved to {self.time_file_name}")

    def log_message(self, message):
        print(message)
        with open(self.log_file_path, "a") as log_file:
            log_file.write(message + "\n")

    def process_and_encode_image(self, image, resize_factor=0.9):
        MAX_SIZE = 20 * 1024 * 1024  # 20MB
        original_width, original_height = image.size

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        for attempt in range(5):
            buffered = io.BytesIO()
            new_width = int(original_width * (resize_factor ** attempt))
            new_height = int(original_height * (resize_factor ** attempt))
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            resized_image.save(buffered, format="JPEG")

            if buffered.tell() < MAX_SIZE:
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                print(f"Attempt {attempt + 1}: Image size is {buffered.tell()} bytes, too large. Resizing...")

        raise ValueError("Unable to reduce image size within 5 attempts")

    def analyze_images_with_Claude_vision(self, prompt_text, encoded_images, temperature=0):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                image_contents = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    } for encoded_image in encoded_images
                ]
                start_time = time.time()
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                *image_contents
                            ],
                        }
                    ],
                    max_tokens=1024,
                    temperature=temperature,
                )
                response_result = response

                if response_result.content[0].text.startswith("I'm sorry, but"):
                    print(f"Response starts with 'I'm sorry', retrying. Attempt {attempt + 1}/{max_attempts}")
                    continue

                end_time = time.time()
                execution_time = end_time - start_time
                self.execution_times.append(execution_time)

                self.print_execution_stats()

                return response_result.content[0].text
            except Exception as e:
                print(f"Error: {e}")
                if "image_parse_error" in str(e).lower() and attempt < max_attempts - 1:
                    resized_encoded_images = []
                    for encoded_image in encoded_images:
                        image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
                        resized_image = self.process_and_encode_image(image, 0.9)
                        resized_encoded_images.append(resized_image)
                    print(f"Adjusting image resolution and retrying. Attempt {attempt + 1}/{max_attempts}")
                    encoded_images = resized_encoded_images
                elif "exceeded" in str(e).lower():
                    self.save_execution_times_to_excel()
                    sys.exit()

        return None

    def print_execution_stats(self):
        total_data_points = len(self.execution_times)
        average_time = sum(self.execution_times) / total_data_points
        max_time = max(self.execution_times)
        min_time = min(self.execution_times)
        std_dev = statistics.stdev(self.execution_times) if total_data_points > 1 else 0

        print(f"Total number of data points: {total_data_points}")
        print(f"Average execution time: {average_time:.2f} seconds")
        print(f"Maximum execution time: {max_time:.2f} seconds")
        print(f"Minimum execution time: {min_time:.2f} seconds")
        print(f"Standard deviation: {std_dev:.2f} seconds")

    def encode_images_from_paths(self, image_paths):
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > 150 and height > 150:
                    encoded_image = self.process_and_encode_image(img)
                    images.append(encoded_image)
        return images

    def create_result_folder(self, base_folder, temperature, try_number):
        folder_name = f"{base_folder}_temp_{str(temperature).replace('.', '_')}_try{try_number}"
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def analyze_cases(self):
        for temperature in self.temperatures:
            for try_number in range(1, self.max_try+1):
                result_folder = self.create_result_folder(self.base_result_folder, temperature, try_number)
                df = pd.read_excel('Lancet_QnA.xlsx')

                for _, row in df.iterrows():
                    case_number = row['no.']
                    file_names = [f"{file_name.strip()}.jpg" for file_name in str(row['jpg']).split(',')]
                    image_paths = [os.path.join(self.case_folder, file_name) for file_name in file_names]
                    print("Filtered image paths:", image_paths)

                    directory_path = os.path.join(result_folder)
                    os.makedirs(directory_path, exist_ok=True)
                    result_file_path = os.path.join(directory_path, f"{case_number}.txt")

                    if self.should_skip_case(case_number, temperature, try_number):
                        continue

                    symptom_text = f"symptom: {row['Q.']}"
                    prompt_text = self.generate_prompt(symptom_text)

                    encoded_images = self.encode_images_from_paths(image_paths)

                    start_time = time.time()
                    result = self.analyze_images_with_Claude_vision(prompt_text, encoded_images, temperature)
                    end_time = time.time()
                    execution_time = end_time - start_time

                    self.update_execution_times(case_number, temperature, try_number, execution_time)

                    if result:
                        self.save_result(result, result_file_path, case_number, temperature, try_number)
                    else:
                        self.log_no_result(case_number, temperature, try_number)

        self.save_execution_times_to_excel()

    def should_skip_case(self, case_number, temperature, try_number):
        if ((self.df_execution_times['number'] == case_number) & 
            (self.df_execution_times['temperature'] == temperature) & 
            (self.df_execution_times['try'] == try_number)).any():
            print(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): skip")
            return True
        return False

    def generate_prompt(self, symptom_text):
        return f"""
        Assignment: You are tasked with solving a quiz on a special medical case involving mostly common diseases. One or more imaging data files will be provided for analysis. The availability of the patient's basic demographic details (age, gender, symptoms) is not guaranteed. The purpose of this assignment is not to provide medical advice or diagnosis but rather to analyze and interpret the imaging data to derive insights related to specified outcomes. This is a purely educational scenario designed for virtual learning situations, aimed at facilitating analysis and educational discussions.

        Your task is to analyze each image individually, or each set of images if multiple types are combined, and derive the following outcomes based on the information provided:

        Outputs:
        1. Type of Medical Imaging: Identify whether the imaging is MR, CT, US, X-ray, Angiography, or Nuclear Medicine. If multiple imaging types are detected in a set, enumerate each type (e.g., "a.MR b.CT c..").
        2. For MR, specify if it's T1WI, T2WI, FLAIR, DWI, SWI, GRE, contrast-enhanced T1WI, TOF, or contrast-enhanced MR angiography. For CT, state whether it is precontrast or postcontrast. For Ultrasound, indicate if it's gray scale or Doppler imaging, etc. Use the format "a.xxx b.xxx c.." if multiple sequences or modes are identified.
        3. Use of Contrast: Note whether a contrast medium was used. Format any multiple entries as "a.Yes b.No c..".
        4. Image Plane: Determine the plane of the image - axial, coronal, sagittal, or other. If multiple planes are evident, list them as "a.axial b.coronal c..".
        5. Specify the body part captured in the imaging. For multiple body parts, use "a.head b.abdomen c..".
        6. Locate the abnormal finding by identifying any unusual or pathological features and noting their anatomical location.

        You are to provide answers in the following JSON format:
        {{
            "1_TypeOfMedicalImaging": "Enter the type or types of imaging used.",
            "2_SpecificImagingSequence": "Specify the sequence or mode used for each type, if multiple.",
            "3_UseOfContrast": "State whether contrast medium was used, format as needed for multiples.",
            "4_ImagePlane": "Mention the plane of the image, list all that apply.",
            "5_PartOfTheBodyImaged": "Identify the body part imaged, enumerate if multiple.",
            "6_LocationOfAbnormalFinding": "Specify the location of the abnormal finding in the image."
        }}
        """

    def update_execution_times(self, case_number, temperature, try_number, execution_time):
        new_row = pd.DataFrame({
            'number': [case_number],
            'temperature': [temperature],
            'try': [try_number],
            'time': [execution_time]
        })
        
        existing_entry = (
            (self.df_execution_times['number'] == case_number) & 
            (self.df_execution_times['temperature'] == temperature) & 
            (self.df_execution_times['try'] == try_number)
        )
        
        if existing_entry.any():
            self.df_execution_times.loc[existing_entry, 'time'] = execution_time
        else:
            self.df_execution_times = pd.concat([self.df_execution_times, new_row], ignore_index=True)
        
        self.df_execution_times = self.df_execution_times.astype({
            'number': 'int',
            'temperature': 'float',
            'try': 'int',
            'time': 'float'
        })

    def save_result(self, result, result_file_path, case_number, temperature, try_number):
        with open(result_file_path, "w") as result_file:
            result_file.write(result)
        print(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): Result has been saved.")

    def log_no_result(self, case_number, temperature, try_number):
        message = f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): No result found."
        print(message)
        self.log_message(message)

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    analyzer = ClaudeVisionAnalyzer(api_key)
    analyzer.analyze_cases()

if __name__ == "__main__":
    main()
