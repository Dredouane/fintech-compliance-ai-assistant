import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

# Configure the SDK with your API key
genai.configure(api_key=api_key)

print("Attempting to connect to the Gemini API...")

try:
    # First, list all available models
    print("Listing available models...")
    for m in genai.list_models():
        print(f"  - {m.name}")

    # Find a model that supports text generation
    found_model_name = None
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            found_model_name = model.name
            break
    
    if not found_model_name:
        raise Exception("No models found that support 'generateContent'.")

    print(f"\nFound a suitable model: {found_model_name}")
    
    # Use the found model for text generation
    model = genai.GenerativeModel(found_model_name)
    response = model.generate_content("What is the capital of France?")
    
    if response and response.text:
        print("Connection successful! Response from Gemini:")
        print(response.text)
    else:
        print("Connection failed: Received an empty or invalid response.")
        
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your API key and network connection. Also, ensure your region supports the Gemini API.")