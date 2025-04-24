import cv2
import models
import torch
import json
import os
import sys
from transformers import pipeline
from datetime import datetime

HISTORY_FILE = "conversation_history.json"
DEFAULT_IMAGE_PATH = "plant2.jpeg"  

def process_text(text):
    processed = text.replace("**", "")
    processed = processed.replace("\\n", "\n")
    return processed

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return None

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def initialize_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=dtype,
        device_map="auto"
    )

def get_plant_info(latin_name, pipe):
    messages = [
        {"role": "system", "content": '''You are a friendly and knowledgeable horticulture expert. 
                The user has uploaded an image of a plant, and the system has identified the plant's scientific name. 
                Your task is to imagine you've seen the image (don't describe or compliment about the plant since you're practically not seeing the image, start with something like 'I see a...') and provide a human-like, conversational response. 
                You should:\n
                1. mention the plant's scientific name and its common/main name, please note that you are suppose to use the plant's common name afterwards and don't use the plant's scientific name.\n
                2. Share care instructions, including:\n
                   - Watering schedule (how often and how much to water).\n
                   - Light requirements (e.g., full sun, partial shade, low light).\n
                   - Common pests and how to address them.\n
                3. Conclude by offering further assistance, such as asking if the user wants to know more about the plant or other related tips.'''
            },
        {"role": "user", "content": latin_name},
    ]
    
    outputs = pipe(messages, max_new_tokens=800)
    return process_text(outputs[0]["generated_text"][-1]['content'])

def process_query(image_path=None, user_query=None):
    """
    Process a single request, either initial plant identification or follow-up question.
    
    Args:
        image_path: Path to plant image (required for first conversation)
        user_query: User's question for follow-up (required for subsequent conversations)
        
    Returns:
        Dict containing assistant's response and status
    """
    history = load_history()
    pipe = initialize_pipeline()
    
    if history is None:
        if image_path is None:
            return {"status": "error", "message": "First conversation requires an image"}
        
        plant_identifier = models.PlantIdentifier()
        image = cv2.imread(image_path)
        
        if image is None:
            return {"status": "error", "message": f"Could not read image from {image_path}"}
        
        outputs = plant_identifier.identify(image, topk=5)
        
        if outputs['status'] != 0:
            return {"status": "error", "message": f"Plant identification failed: {outputs['message']}"}
        
        first_latin_name = outputs['results'][0]['latin_name']
        confidence = outputs['results'][0]['probability']
        plant_info = get_plant_info(first_latin_name, pipe)
        
        history = {
            "plant_latin_name": first_latin_name,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "conversation": [
                {"role": "system", "content": f"Plant identified as {first_latin_name}"},
                {"role": "assistant", "content": plant_info}
            ]
        }
        save_history(history)
        
        return {
            "status": "success",
            "plant": first_latin_name,
            "confidence": confidence,
            "response": plant_info
        }
    
    else:
        if user_query is None:
            return {"status": "error", "message": "Follow-up conversation requires a query"}
        
        plant_latin_name = history["plant_latin_name"]
        
        messages = [
            {"role": "system", "content": "You are a friendly and knowledgeable horticulture expert."},
            {"role": "user", "content": f"About {plant_latin_name}: {user_query}"}
        ]
        
        for msg in history["conversation"]:
            if msg["role"] != "system":  
                messages.insert(1, msg) 
        
        outputs = pipe(messages, max_new_tokens=800)
        assistant_response = process_text(outputs[0]["generated_text"][-1]['content'])
        
        history["conversation"].append({"role": "user", "content": user_query})
        history["conversation"].append({"role": "assistant", "content": assistant_response})
        save_history(history)
        
        return {
            "status": "success",
            "response": assistant_response
        }

if __name__ == "__main__":
    if len(sys.argv) == 1:
        image_path = DEFAULT_IMAGE_PATH
        
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            sys.exit(1)
        
        clear_history()
        
        print(f"\nProcessing plant image '{image_path}', please wait...")
        result = process_query(image_path=image_path)
        
        if result["status"] == "success":
            print(f"\nPlant identified as: {result['plant']} (Confidence: {result['confidence']:.2%})")
            print("\nPlant Information:")
            print(result["response"])

        else:
            print(f"Error: {result['message']}")
            sys.exit(1)
    else:
        user_query = " ".join(sys.argv[1:])
        
        if not load_history():
            sys.exit(1)
            
        result = process_query(user_query=user_query)
        
        if result["status"] == "success":
            print("\nResponse:")
            print(result["response"])
        else:
            print(f"Error: {result['message']}")
            sys.exit(1)