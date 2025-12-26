"""
JSON Parser Utility
ICDAM 2025 - Negotiation Layer Parsing
"""

import json
import re
from typing import Dict, Any, Optional

class JSONParser:
    """
    Utility class to parse structured JSON from LLM responses.
    Handles markdown blocks and loose formatting.
    """
    
    @staticmethod
    def parse_llm_response(text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response text.
        Looks for thought, speak, and function fields.
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Dict containing parsed fields or default values.
        """
        # Default structure
        result = {
            "thought": "",
            "speak": "",
            "function": None,
            "raw": text
        }
        
        try:
            # 1. Try to find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 2. Try to find anything that looks like a JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = text
            
            parsed = json.loads(json_str)
            
            # Update result with parsed values
            if isinstance(parsed, dict):
                result.update(parsed)
            elif isinstance(parsed, list):
                # If it's a list, it's likely the "function" or data part
                result["function"] = parsed
            
        except (json.JSONDecodeError, Exception):
            # Fallback: try to extract fields manually if JSON parsing fails
            # This is a very basic fallback
            thought_match = re.search(r'"thought":\s*"(.*?)"', text)
            if thought_match:
                result["thought"] = thought_match.group(1)
                
            speak_match = re.search(r'"speak":\s*"(.*?)"', text)
            if speak_match:
                result["speak"] = speak_match.group(1)
            else:
                # If no speak field, use the whole text as speak content (minus JSON-like parts)
                result["speak"] = text[:200] # Limit size
                
        return result
