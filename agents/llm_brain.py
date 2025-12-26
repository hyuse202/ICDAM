"""
LLM Brain Module - Robust Hybrid Client
Connects the Multi-Agent System to Google Gemini API with intelligent fallback.
Features graceful degradation to mock responses when API fails.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv


class LLMBrain:
    """
    Robust Hybrid LLM Client with automatic fallback to mock responses.
    Ensures uninterrupted operation despite API errors (429, 404, network issues).
    """
    
    def __init__(self):
        """
        Initialize the LLM Brain with Google Gemini.
        Loads API credentials and configures the model.
        """
        # Load environment variables
        load_dotenv()
        
        # Retrieve configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("LLM_MODEL_NAME", "models/gemini-2.5-flash")
        
        self.model_name = model_name
        self.api_available = False
        self.model = None
        
        # Attempt to configure the API
        if not api_key:
            print("[WARNING] GOOGLE_API_KEY not found in environment")
            print("[INFO] Running in MOCK MODE only")
            return
        
        try:
            # Configure the API
            genai.configure(api_key=api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(model_name)
            self.api_available = True
            
            print(f"[INFO] LLM Brain initialized (Model: {self.model_name})")
            print("[INFO] Fallback Mode: Enabled (auto-switches on API errors)")
        
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM API: {type(e).__name__}")
            print("[INFO] Running in MOCK MODE only")
    
    def _get_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response based on keywords in the prompt.
        Used as fallback when API is unavailable.
        
        Args:
            prompt: The original prompt text
            
        Returns:
            str: Simulated response based on prompt content
        """
        prompt_lower = prompt.lower()
        
        # Keyword-based mock responses
        if "connected" in prompt_lower or "hello" in prompt_lower:
            return "CONNECTED (MOCK MODE)"
        
        elif "book" in prompt_lower or "resource" in prompt_lower or "allocate" in prompt_lower:
            return "AGREED. Resources provided."
        
        elif "plan" in prompt_lower or "schedule" in prompt_lower or "time" in prompt_lower:
            return "PROPOSAL. Duration: 45 days."
        
        elif "status" in prompt_lower or "report" in prompt_lower:
            return "STATUS: All systems operational. Working in offline mode."
        
        else:
            # Generic fallback
            return "MOCK RESPONSE: I heard you."
    
    def think(self, prompt: str) -> str:
        """
        Send a prompt to the LLM with automatic fallback on errors.
        
        Args:
            prompt: Text prompt to send to the LLM
            
        Returns:
            str: LLM's text response, or mock response if API fails
        """
        # If model was not initialized, use mock immediately
        if self.model is None:
            return self._get_mock_response(prompt)
        
        try:
            # Attempt to call the real API
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            if hasattr(response, 'text'):
                self.api_available = True
                return response.text
            else:
                # Response exists but no text - try to get it another way
                if hasattr(response, 'parts') and len(response.parts) > 0:
                    return str(response.parts[0].text)
                else:
                    raise ValueError("No text content in response")
        
        except Exception as e:
            # API call failed - use fallback
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Print warning
            print(f"[WARNING] API Error: {error_type} - {error_msg[:80]}")
            print("[WARNING] Switching to Mock response...")
            
            # Mark API as unavailable and return mock response
            self.api_available = False
            return self._get_mock_response(prompt)
    
    def think_with_context(self, system_context: str, user_prompt: str) -> str:
        """
        Send a prompt with system context to the LLM.
        
        Args:
            system_context: Background context/instructions for the LLM
            user_prompt: User's specific question or request
            
        Returns:
            str: LLM's text response or mock response
        """
        # Combine context and prompt
        full_prompt = f"{system_context}\n\n{user_prompt}"
        return self.think(full_prompt)
    
    def is_api_available(self) -> bool:
        """
        Check if the API is currently available.
        
        Returns:
            bool: True if last API call succeeded, False otherwise
        """
        return self.api_available


def main():
    """Test the LLM Brain connection with fallback capability."""
    print("=" * 70)
    print("LLM BRAIN - Robust Hybrid Client Test")
    print("=" * 70)
    
    try:
        # Initialize the brain
        print("\n[1] Initializing LLM Brain...")
        brain = LLMBrain()
        
        # Test prompt
        print("\n[2] Sending test prompt...")
        test_prompt = "Hello Gemini, return only the word 'CONNECTED' if you hear me."
        print(f"    Prompt: \"{test_prompt}\"")
        
        # Get response (will auto-fallback if API fails)
        print("\n[3] Waiting for response...")
        response = brain.think(test_prompt)
        
        # Display result
        print("\n[4] Response received:")
        print("-" * 70)
        print(response)
        print("-" * 70)
        
        # Verify connection
        if "CONNECTED" in response.upper():
            if "MOCK" in response:
                print("\n[SUCCESS] LLM Test: MOCK MODE - API unavailable")
            else:
                print("\n[SUCCESS] LLM Test: REAL API")
        else:
            print("\n[WARNING] LLM Test: Unexpected response format")
        
        # Additional tests with different keywords
        print("\n" + "=" * 70)
        print("[5] Testing Fallback with Different Keywords")
        print("=" * 70)
        
        test_prompts = [
            "Can I book 3 units of resource R1?",
            "What's the optimal schedule for this project?",
            "Give me a status report."
        ]
        
        for idx, prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {idx}: \"{prompt}\"")
            result = brain.think(prompt)
            print(f"  Response: {result}")
        
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE - System remains operational regardless of API status")
    print("=" * 70)


if __name__ == "__main__":
    main()