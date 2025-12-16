"""
Check Available Gemini Models
Diagnostic script to list all models available with your API key.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv


def main():
    """List all available Gemini models that support text generation."""
    print("=" * 70)
    print("GEMINI API - Model Availability Check")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Validate API key
    if not api_key:
        print("\n Error: GOOGLE_API_KEY not found in environment variables")
        print("Please check your .env file")
        return
    
    print(f"\n[1] API Key loaded: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("[2] API configured successfully")
        
        # List all models
        print("\n[3] Fetching available models...")
        models = genai.list_models()
        
        # Filter for models that support generateContent
        print("\n" + "=" * 70)
        print("AVAILABLE MODELS (Supporting 'generateContent')")
        print("=" * 70)
        
        supported_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                supported_models.append(model)
        
        if not supported_models:
            print("\n No models found that support 'generateContent'")
            return
        
        print(f"\nFound {len(supported_models)} models:\n")
        
        for idx, model in enumerate(supported_models, 1):
            print(f"{idx}. Display Name: {model.display_name}")
            print(f"   Model Name:   {model.name}")
            print(f"   Description:  {model.description[:100]}..." if len(model.description) > 100 else f"   Description:  {model.description}")
            print()
        
        # Provide recommendations
        print("=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print("\nCopy one of the 'Model Name' values above to your .env file.")
        print("Example .env configuration:\n")
        
        if supported_models:
            # Show the first model as an example
            example_model = supported_models[0].name
            print(f"LLM_MODEL_NAME={example_model}")
        
        print("\nCommon options:")
        for model in supported_models:
            if 'flash' in model.name.lower():
                print(f"  - {model.name}  (Fast, efficient)")
            elif 'pro' in model.name.lower():
                print(f"  - {model.name}  (More capable)")
        
    except Exception as e:
        print(f"\n Error occurred while fetching models:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\nPossible causes:")
        print("  - Invalid API key")
        print("  - Network connection issues")
        print("  - API service unavailable")
    
    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
