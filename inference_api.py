import cv2
import models
import json
import os
import sys
import base64
from datetime import datetime
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

HISTORY_FILE = "conversation_history.json"
DEFAULT_IMAGE_PATH = "plant.jpeg"

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

def initialize_client():
    """Initialize the Azure API client for Llama model"""
    if github_token := os.environ.get("GITHUB_TOKEN"):
        return ChatCompletionsClient(
            endpoint="https://models.inference.ai.azure.com",
            credential=AzureKeyCredential(github_token),
        )
    else:
        raise EnvironmentError("GITHUB_TOKEN environment variable not set. Please set your GitHub Personal Access Token.")

def get_plant_info(latin_name, client):
    """Get plant information using the Azure API client"""
    system_prompt = """You are a friendly and knowledgeable horticulture expert. 
    The user has uploaded an image of a plant, and the system has identified the plant's scientific name. 
    Your task is to imagine you've seen the image (don't describe or compliment about the plant since you're practically not seeing the image, start with something like 'I see a...') and provide a human-like, conversational response. 
    You should:
    1. Mention the plant's scientific name and its common/main name, please note that you are supposed to use the plant's common name afterwards and don't use the plant's scientific name.
    2. Share care instructions, including:
       - Watering schedule (how often and how much to water).
       - Light requirements (e.g., full sun, partial shade, low light).
       - Common pests and how to address them.
    3. Conclude by offering further assistance, such as asking if the user wants to know more about the plant or other related tips."""

    response = client.complete(
        messages=[
            SystemMessage(system_prompt),
            UserMessage(f"The plant has been identified as {latin_name}. Please provide care information about it.")
        ],
        model="Llama-3.2-11B-Vision-Instruct",
        temperature=0.7,
        max_tokens=800,
    )
    
    return response.choices[0].message.content

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
    client = initialize_client()
    
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
        plant_info = get_plant_info(first_latin_name, client)
        
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
        
        azure_messages = [SystemMessage("You are a friendly and knowledgeable horticulture expert.")]
        
        for msg in history["conversation"]:
            if msg["role"] == "system":
                continue 
            elif msg["role"] == "user":
                azure_messages.append(UserMessage(msg["content"]))
            elif msg["role"] == "assistant":
                azure_messages.append(AssistantMessage(msg["content"]))
        
        azure_messages.append(UserMessage(f"About {plant_latin_name}: {user_query}"))
        
        response = client.complete(
            messages=azure_messages,
            model="Llama-3.2-11B-Vision-Instruct",
            temperature=0.7,
            max_tokens=800,
        )
        
        assistant_response = response.choices[0].message.content
        
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
            print("Error: No plant has been identified yet. Please run without arguments first.")
            sys.exit(1)
            
        result = process_query(user_query=user_query)
        
        if result["status"] == "success":
            print("\nResponse:")
            print(result["response"])
        else:
            print(f"Error: {result['message']}")
            sys.exit(1)