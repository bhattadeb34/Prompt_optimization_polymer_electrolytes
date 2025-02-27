from typing import Dict
import langfun as lf

class LLMConfig:
    """Configuration for LLM models"""
    
    MODEL_CONFIGS = {
        "gpt-4": {
            "api_key_name": "openai",
            "model": lambda key, temp, tokens: lf.llms.openai.Gpt4(
                api_key=key,
                temperature=temp,
                max_tokens=tokens
            )
        },
        "claude-3-opus": {
            "api_key_name": "claude",
            "model": lambda key, temp, tokens: lf.llms.anthropic.Claude3Opus(
                api_key=key,
                temperature=temp,
                max_tokens=tokens
            )
        },
        "claude-3-5-sonnet": {
            "api_key_name": "claude",
            "model": lambda key, temp, tokens: lf.llms.anthropic.Claude35Sonnet(
                api_key=key,
                temperature=temp,
                max_tokens=tokens
            )
        },
        "claude-3-haiku": {
            "api_key_name": "claude",
            "model": lambda key, temp, tokens: lf.llms.anthropic.Claude3Haiku(
                api_key=key,
                temperature=temp,
                max_tokens=tokens
            )
        }
    }
    
    @classmethod
    def get_model(cls, 
                 model_name: str, 
                 api_keys: Dict[str, str],
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        """Get LLM model instance"""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODEL_CONFIGS.keys())}")
        
        config = cls.MODEL_CONFIGS[model_name]
        api_key_name = config["api_key_name"]
        
        if api_key_name not in api_keys:
            raise ValueError(f"No API key found for {api_key_name}")
            
        return config["model"](api_keys[api_key_name], temperature, max_tokens)