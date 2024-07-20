import os
import json
import openai
import io
import base64
from PIL import Image
import time
import statistics
import pandas as pd
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.getcwd())
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)
os.environ['OPENAI_API_KEY'] = os.getenv("MY_OPENAI_API_KEY")

class GPT4VisionAnalyzer:
    def __init__(self, api_key, time_file_name="OpenAI_gpt4o_execution_times.xlsx"):
        self.api_key = api_key
        self.execution_times = []
        self.time_file_name = os.path.join('time', time_file_name)
        self.df_execution_times = self.load_or_initialize_execution_times()
        self.log_file_path = os.path.join("./", "process_log.txt")
        self.temperatures = [0]
        self.base_result_folder = "gpt4o_result/gpt4o_result"
        self.max_try = 1
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    def load_or_initialize_execution_times(self):
        if os.path.exists(self.time_file_name):
            print("Loading time file")
            return pd.read_excel(self.time_file_name)
        else:
            print("Creating time file")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.time_file_name), exist_ok=True)
            return pd.DataFrame({
                'number': pd.Series(dtype='int'),
                'temperature': pd.Series(dtype='float'),
                'try': pd.Series(dtype='int'),
                'time': pd.Series(dtype='float')
            })

    def save_execution_times_to_excel(self):
        os.makedirs(os.path.dirname(self.time_file_name), exist_ok=True)
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
                print(f"Attempt {attempt + 1}: Image too large. Resizing...")

        raise ValueError("Unable to reduce image size within 5 attempts")

    def analyze_images_with_gpt4_vision(self, prompt_text, encoded_images, temperature=0):
        client = openai.OpenAI()
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                image_contents = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    } for img in encoded_images
                ]
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                *image_contents
                            ]
                        }
                    ],
                    max_tokens=1024,
                    temperature=temperature,
                )
                response_result = response.choices[0]

                if response_result.message.content.startswith("I'm sorry, but"):
                    print(f"I'm sorry response. Retrying {attempt + 1}/{max_attempts}")
                    continue

                end_time = time.time()
                execution_time = end_time - start_time
                self.execution_times.append(execution_time)

                self.print_execution_stats()

                return response_result
            except Exception as e:
                print(f"BadRequestError: {e}")
                if "image_parse_error" in str(e).lower() and attempt < max_attempts - 1:
                    resized_encoded_images = []
                    for encoded_image in encoded_images:
                        image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
                        resized_image = self.process_and_encode_image(image, 0.9)
                        resized_encoded_images.append(resized_image)
                    print(f"Resizing image and retrying {attempt + 1}/{max_attempts}")
                    encoded_images = resized_encoded_images

        return None

    def print_execution_stats(self):
        avg_time = sum(self.execution_times) / len(self.execution_times)
        max_time = max(self.execution_times)
        min_time = min(self.execution_times)
        std_dev = statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0
        total_points = len(self.execution_times)

        print(f"Total data points: {total_points}")
        print(f"Average execution time: {avg_time:.2f} seconds")
        print(f"Max execution time: {max_time:.2f} seconds")
        print(f"Min execution time: {min_time:.2f} seconds")
        print(f"Standard deviation: {std_dev:.2f} seconds")

    def read_text_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def encode_images_from_paths(self, image_paths):
        images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Error: Image file does not exist: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width > 150 and height > 150:
                        encoded_image = self.process_and_encode_image(img)
                        images.append(encoded_image)
                        print(f"Successfully encoded image: {image_path}")
                    else:
                        print(f"Image too small, skipping: {image_path}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        return images

    def create_result_folder(self, base_folder, temperature, try_number):
        folder_name = (
            f"{base_folder}_temp_{str(temperature).replace('.', '_')}"
            f"_try{try_number}"
        )
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def analyze_cases(self):
        for temperature in self.temperatures:
            for try_number in range(1, self.max_try + 1):
                result_folder = self.create_result_folder(
                    self.base_result_folder, temperature, try_number
                )
                results_df = pd.DataFrame(
                    columns=['case_number', '1_TypeOfMedicalImaging', '2_SpecificImagingSequence',
                             '3_UseOfContrast', '4_ImagePlane', '5_PartOfTheBodyImaged', '6_LocationOfAbnormalFinding']
                )
                df = pd.read_excel('Lancet_QnA.xlsx')

                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                case_folder = os.path.join(parent_dir, "Lancet_IMAGE240508")

                for _, row in df.iterrows():
                    case_number = row['no.']
                    file_names = [
                        file_name.strip() for file_name in str(row['jpg']).split(',')
                    ]
                    
                    image_paths = self.get_image_paths(case_folder, file_names)

                    print("Filtered image paths:", image_paths)

                    directory_path = os.path.join(result_folder)
                    os.makedirs(directory_path, exist_ok=True)
                    result_file_path = os.path.join(
                        directory_path, f"{case_number}.txt"
                    )

                    if self.should_skip_case(case_number, temperature, try_number):
                        continue

                    symptom_text = f"symptom: {row['Q.']}"
                    prompt_text = self.generate_prompt(symptom_text)

                    print(image_paths)
                    encoded_images = self.encode_images_from_paths(image_paths)

                    start_time = time.time()
                    result = self.analyze_images_with_gpt4_vision(
                        prompt_text, encoded_images, temperature
                    )
                    end_time = time.time()
                    execution_time = end_time - start_time

                    self.update_execution_times(
                        case_number, temperature, try_number, execution_time
                    )

                    if result:
                        self.save_result(
                            result, result_file_path, case_number,
                            temperature, try_number
                        )
                        results_df = self.update_results_df(results_df, case_number, result)
                    else:
                        self.log_no_result(case_number, temperature, try_number)

                    self.save_results_to_excel(results_df, result_folder)

        self.save_execution_times_to_excel()

    def get_image_paths(self, case_folder, file_names):
        image_paths = []
        for file_name in file_names:
            file_path = None
            if os.path.exists(os.path.join(case_folder, file_name)):
                file_path = os.path.join(case_folder, file_name)
            else:
                for ext in self.image_extensions:
                    potential_path = os.path.join(case_folder, f"{file_name}{ext}")
                    if os.path.exists(potential_path):
                        file_path = potential_path
                        break
            
            if file_path:
                image_paths.append(file_path)
            else:
                print(f"Warning: Image file not found for {file_name} in {case_folder}")
        return image_paths

    def should_skip_case(self, case_number, temperature, try_number):
        if ((self.df_execution_times['number'] == case_number) & 
            (self.df_execution_times['temperature'] == temperature) & 
            (self.df_execution_times['try'] == try_number)).any():
            print(f"Case {case_number} (Temperature: {temperature}, "
                  f"Try: {try_number}): skip")
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
            # Ensure all columns are present in new_row
            for col in self.df_execution_times.columns:
                if col not in new_row.columns:
                    new_row[col] = pd.NA
            self.df_execution_times = pd.concat([self.df_execution_times, new_row], ignore_index=True)

    def save_result(self, result, result_file_path, case_number, temperature, try_number):
        with open(result_file_path, "w") as result_file:
            result_file.write(result.message.content)
        print(f"Case {case_number} (Temperature: {temperature}, "
            f"Try: {try_number}): Result saved.")

    def update_results_df(self, results_df, case_number, result):
        result_content = json.loads(result.message.content)
        new_row = pd.DataFrame({
            'case_number': [case_number],
            '1_TypeOfMedicalImaging': [result_content['1_TypeOfMedicalImaging']],
            '2_SpecificImagingSequence': [result_content['2_SpecificImagingSequence']],
            '3_UseOfContrast': [result_content['3_UseOfContrast']],
            '4_ImagePlane': [result_content['4_ImagePlane']],
            '5_PartOfTheBodyImaged': [result_content['5_PartOfTheBodyImaged']],
            '6_LocationOfAbnormalFinding': [result_content['6_LocationOfAbnormalFinding']]
        })
        return pd.concat([results_df, new_row], ignore_index=True)

    def log_no_result(self, case_number, temperature, try_number):
        message = (f"Case {case_number} (Temperature: {temperature}, "
                   f"Try: {try_number}): No result found.")
        print(message)
        self.log_message(message)

    def save_results_to_excel(self, results_df, result_folder):
        excel_path = os.path.join(result_folder, 'analysis_results.xlsx')
        results_df.to_excel(excel_path, index=False)
        print("Results have been saved to 'analysis_results.xlsx'.")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    analyzer = GPT4VisionAnalyzer(api_key)
    analyzer.analyze_cases()

if __name__ == "__main__":
    main()
