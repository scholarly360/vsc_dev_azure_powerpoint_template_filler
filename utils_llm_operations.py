import os
from openai import AzureOpenAI, azure_endpoint
from dotenv import load_dotenv
import os
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import json
import re

# Load environment variables from .env file
load_dotenv()

from CustomSemanticMatcher import CustomSemanticMatcher



def summarize_text(custom_para_text: str) -> str:
    """
    Summarizes the given text using Azure OpenAI.

    AZURE_OPENAI_API_KEY= 
    AZURE_OPENAI_API_VERSION=2024-08-01-preview
    AZURE_OPENAI_ENDPOINT=https://vedabase-io-west.openai.azure.com/
    AZURE_MODEL=gpt-4o-mini
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_MODEL", "gpt-4o-mini")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Initialize Azure OpenAI Service client with key-based authentication
    az_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )

    custom_system_instruction = """
    You are a helpful assistant that summarizes text. Please provide a concise summary of the input text.
    """
    chat_prompt = [
        {
            "role": "system", "content": [{"type": "text", "text": custom_system_instruction}],
        },
        {
            "role": "user", "content": [{"type": "text", "text": custom_para_text}],
        }
    ]

    # Include speech result if speech is enabled
    messages = chat_prompt

    # Generate the completion
    completion = az_client.chat.completions.create(model=deployment,
                                                   messages=messages,
                                                   max_tokens=200,
                                                   temperature=0.1,
                                                   top_p=0.95,
                                                   frequency_penalty=0,
                                                   presence_penalty=0,
                                                   stop=None,
                                                   stream=False,
                                                   )
    return (completion.to_dict()["choices"][0]['message']['content'])


def match_slides_and_text(list_slides, list_summary_text):
    """
    Matches slides with their corresponding summary text.


    list_slides: List of slide information dictionaries.
    list_summary_text: List of summary text strings.

    [

        {'slide_number': 1,
        'slide_layout_type': 'Two Content',
        'slide_layout_type_content': True,
        'slide_list_text': 
            [{'run_id': 3,
            'run_font': None,
            'run_text': 'Definition and Purpose of an RFP',
            'run_text_len': 32},
            {'run_id': 6,
                'run_font': 177800,
                'run_text': 'What is an RFP',
                'run_text_len': 14}]
    ]


    """

    # Step 1 : Only consider if 'slide_layout_type_content': True,
    filtered_slides = [
        slide for slide in list_slides if slide['slide_layout_type_content']]

    # Step 2 : Make consolidated text for each slide

    list_slide_summaries = []
    for slide in filtered_slides:
        slide_text = ""
        for text in slide['slide_list_text']:
            slide_text += text['run_text'] + " "
        list_slide_summaries.append(
            {'slide_number': slide['slide_number'], 'text': slide_text.strip()})
        

    # Step 3 : Match with summary text

    obj_semantic_matcher = CustomSemanticMatcher()
    list_semantic_matches = obj_semantic_matcher.find_semantic_matches(list_slide_summaries, list_summary_text, top_k=1)

    """
        [
            {
                'slide_number': 3,
                'slide_text': '',
                'matched_summary_id': 0,
                'matched_summary_text': "",
            }
        ]
    
    """

    """ return origional and matched slides """
    return (list_slides, list_semantic_matches)



def post_matching_generate(list_slides, list_semantic_matches):
    """
    list_slides: List of slide information dictionaries.
    list_semantic_matches: List of semantic match dictionaries.

    [

        {'slide_number': 1,
        'slide_layout_type': 'Two Content',
        'slide_layout_type_content': True,
        'slide_list_text': 
            [{'run_id': 3,
            'run_font': None,
            'run_text': 'Definition and Purpose of an RFP',
            'run_text_len': 32},
            {'run_id': 6,
                'run_font': 177800,
                'run_text': 'What is an RFP',
                'run_text_len': 14}]
    ]



    list_semantic_matches ::
        [
            {
                'slide_number': 3,
                'slide_text': '',
                'matched_summary_id': 0,
                'matched_summary_text': "",
            }
        ]
    """

