# Lancet-PictureQuizzes-VLM-Challenge
The Lancet  “Picture Quiz Gallery” - VLM Image Challenge


## Abstract

This study aimed to assess the ability of six different multimodal Large Language Models (LLMs)—GPT-4 Turbo with Vision (GPT-4v), GPT-4 Omni (GPT-4o), Gemini 1.5 Pro, Gemini Flash, Claude 3.0, and Claude 3.5—in solving picture quizzes from The Lancet with radiologic images. The focus was on the independence of test data and the stochasticity of LLMs.

A search of cases was conducted on the “Picture Quiz Gallery” of The Lancet website (https://www.thelancet.com/picture-quiz) in May 2024. Among 431 quiz cases from January 2012 to March 2024, cases with radiologic images composed of X-ray, CT, MRI, US, angiography, and fluoroscopy were included in this study. Cases with only clinical photos were excluded (n = 204). Texts consisting of multiple-choice questions were extracted using copy and paste, and images were downloaded from the website.

**Note**: This study is currently under publication.

## Project Structure

```
Lancet-PictureQuizzes-VLM-Challenge
├── 1_SolvingQuiz_Task
│   ├── 1.1.1.gpt-4-turbo_orig.py
│   ├── 1.1.2.gpt-4o_orig.py
│   ├── 1.1.3.gpt-4-turbo_rephrased.py
│   ├── 1.1.4.gpt-4o_rephrased.py
│   ├── 1.2.1.gemini-1.5-pro_orig.py
│   ├── 1.2.2.gemini-1.5-flash_orig.py
│   ├── 1.2.3.gemini-1.5-pro_rephrased.py
│   ├── 1.2.4.gemini-1.5-flash_rephrased.py
│   ├── 1.3.1.claude-3-opus_orig.py
│   ├── 1.3.2.claude-3-5-sonnet_orig.py
│   ├── 1.3.3.claude-3-opus_rephrased.py
│   ├── 1.3.4.claude-3-5-sonnet_rephrased.py
│   ├── 1.4.excel_combined_sum.py
├── 2_VisionModel_Med-Task
│   ├── 2.1.1.gpt-4-turbo_describe.py
│   ├── 2.1.2.gpt-4o_describe.py
│   ├── 2.2.1.gemini-1.5-pro_describe.py
│   ├── 2.2.2.gemini-1.5-flash_describe.py
│   ├── 2.3.1.claude-3-opus_describe.py
│   ├── 2.3.2.claude-3-5-sonnet_describe.py
│   ├── 2.4.excel_combined_sum.py
├── 3_Image-Removed_Task
│   ├── 3.1.3.gpt-4-turbo_rephrased_img-removed.py
│   ├── 3.1.4.gpt-4o_rephrased_img-removed.py
│   ├── 3.2.3.gemini-1.5-pro_rephrased_img-removed.py
│   ├── 3.2.4.gemini-1.5-flash_rephrased_img-removed.py
│   ├── 3.3.3.claude-3-opus_rephrased_img-removed.py
│   ├── 3.3.4.claude-3-5-sonnet_rephrased_img-removed.py
│   ├── 3.4.excel_combined_sum.py
├── Lancet_QnA.xlsx
├── requirements.txt
├── dot_env_file_here.env
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install required dependencies

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Core-BMC/Lancet-PictureQuizzes-VLM-Challenge.git
cd Lancet-PictureQuizzes-VLM-Challenge
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

   Rename `dot_env_file_here.env` to `.env` and add your API keys.

### Usage

1. **Solving Quiz Task**:
   - Original and rephrased scripts for GPT-4 Turbo, GPT-4 Omni, Gemini 1.5 Pro, Gemini Flash, Claude 3.0, and Claude 3.5 models.
   - Scripts located in the `1_SolvingQuiz_Task` folder.
   - Run each script for analyzing images and generating results.

2. **Vision Model Medical Task**:
   - Description-based analysis scripts for the models.
   - Located in the `2_VisionModel_Med-Task` folder.
   - Run these scripts for generating detailed analysis based on descriptions.

3. **Image-Removed Task**:
   - Analysis scripts without images.
   - Located in the `3_Image-Removed_Task` folder.
   - Run these scripts for generating text-based outputs.

4. **Combining Results**:
   - Scripts for combining results into a single Excel file.
   - Located in each task folder and the root directory.
   - Run the `excel_combined_sum.py` script in each folder to consolidate results.

### Example Commands

To run a specific analysis script:

```bash
python 1_SolvingQuiz_Task/1.1.1.gpt-4-turbo_orig.py
```

To combine results into an Excel file:

```bash
python 1_SolvingQuiz_Task/1.4.excel_combined_sum.py
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

**Woo Hyun Shim** - [swh@amc.seoul.kr](mailto:swh@amc.seoul.kr)

[Hwon Heo](https://github.com/hwonheo) - [heohwon@gmail.com](mailto:heohwon@gmail.com)

Project Link: [https://github.com/Core-BMC/Lancet-PictureQuizzes-VLM-Challenge](https://github.com/Core-BMC/Lancet-PictureQuizzes-VLM-Challenge)
