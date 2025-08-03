import os
import traceback
import json
from doc_insights_utils import * 
from utils_llm_operations import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if(False):
    try:
        # Check if input file exists
        input_file = './input_docs/sample-contract.pdf'
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
        else:
            print(f"Input file exists: {input_file}")
        
        # Check if output directory exists, create if not
        output_dir = './tmp_folder'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        doc_to_png(input_file, output_dir)

        list_png_files = get_png_files('./tmp_folder')
        print(list_png_files)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are installed")

doc_client = get_client()
json_output_file = './output_text_data.json'

# list_txt_doc_files = analyze_files_in_directory(doc_client, './tmp_folder')
# list_txt_only = get_page_wise_text(list_txt_doc_files)
# with open(json_output_file, 'w', encoding='utf-8') as f:
#     json.dump(list_txt_only, f, indent=2, ensure_ascii=False)
# print(f"Saved text data to: {json_output_file}")

# Read from JSON file
with open(json_output_file, 'r', encoding='utf-8') as f:
    list_loaded_data = json.load(f)

print("Loaded data from JSON:")
print(json.dumps(list_loaded_data, indent=2))

list_summary_text = []
for index, item in enumerate(list_loaded_data):
    summary = summarize_text(item["full_content"])
    list_summary_text.append({"id": index, "summary": summary})
    print("-------")


json_output_summary_file = './output_text_summary.json'
with open(json_output_summary_file, 'w', encoding='utf-8') as f:
    json.dump(list_summary_text, f, indent=2, ensure_ascii=False)
print(f"Saved summary data to: {json_output_summary_file}")

# Read from JSON file
with open(json_output_summary_file, 'r', encoding='utf-8') as f:
    list_loaded_summary_data = json.load(f)

print("Loaded data from JSON:")
print(json.dumps(list_loaded_summary_data, indent=2))
# delete_png_files(directory='./tmp_folder')