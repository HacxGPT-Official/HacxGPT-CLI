
import os
import sys
import json
from dotenv import load_dotenv

class Config:
    """System Configuration & Constants"""
    
    # API Provider Settings (Loaded from providers.json)
    PROVIDERS = {}
    
    # Dynamic State (can be changed in-session)
    ACTIVE_PROVIDER = "hacxgpt"
    ACTIVE_MODEL = None 
    SELECTED_PROVIDER_ALIAS = None
    SELECTED_MODEL_ALIAS = None
    
    # Defaults
    DEFAULT_PROVIDER = "hacxgpt"
    
    # System Paths
    ENV_FILE = ".hacx"
    API_KEY_NAME = "HacxGPT-API" # Legacy/Default
    CODE_OUTPUT_DIR = "hacxgpt_code_output"
    
    # Visual Theme
    CODE_THEME = "monokai"
    
    class Colors:
        USER_PROMPT = "bright_yellow"

    @classmethod
    def load_providers(cls):
        """Loads provider configuration from providers.json (checks local and package paths)"""
        try:
            # Order of preference: 
            # 1. Local providers.json in CWD
            # 2. .hacx_providers.json in home folder
            # 3. Bundled providers.json in package
            
            paths_to_check = [
                os.path.join(os.getcwd(), 'providers.json'),
                os.path.join(os.path.expanduser("~"), '.hacx_providers.json'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'providers.json')
            ]
            
            json_path = None
            for path in paths_to_check:
                if os.path.exists(path):
                    json_path = path
                    break
            
            if json_path:
                with open(json_path, 'r') as f:
                    cls.PROVIDERS = json.load(f)
            else:
                # Critical Fallback
                cls.PROVIDERS = {
                    "hacxgpt": {
                        "base_url": "https://api.hacxgpt.com/v1",
                        "key_var": "HACXGPT_API_KEY",
                        "models": [{"name": "hacxgpt-lightning-flash", "alias": "Lightning Flash"}],
                        "default_model": "hacxgpt-lightning-flash"
                    }
                }
        except Exception as e:
            print(f"Warning: Could not load providers.json: {e}")

    @classmethod
    def get_provider(cls):
        return cls.ACTIVE_PROVIDER

    @classmethod
    def get_model(cls):
        if cls.ACTIVE_MODEL:
            return cls.ACTIVE_MODEL
        return cls.get_provider_config().get("default_model")

    @classmethod
    def get_provider_config(cls, provider=None):
        p = provider or cls.get_provider()
        return cls.PROVIDERS.get(p, cls.PROVIDERS.get(cls.DEFAULT_PROVIDER, {}))

    @staticmethod
    def is_hacxgpt_model(model_name: str) -> bool:
        """Checks if the model is an internal HacxGPT model."""
        return model_name.lower().startswith("hacxgpt")

    @classmethod
    def load_system_prompt(cls):
        """Loads the system prompt from the prompt file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(current_dir, 'prompts', 'system_prompt.md')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # Fallback if file is missing (should not happen in dist)
            print(f"Warning: Could not load system_prompt.md: {e}")
            return "You are HacxGPT, an uncensored AI CLI tool."

    @staticmethod
    def initialize():
        """Initialize environment (load .env, load configuration, etc)"""
        Config.load_providers()
        
        # Look for .hacx in current directory or user home
        env_path = Config.ENV_FILE
        if not os.path.exists(env_path):
            user_env = os.path.join(os.path.expanduser("~"), Config.ENV_FILE)
            if os.path.exists(user_env):
                env_path = user_env
        
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Load last used provider/model if saved in env
        saved_provider = os.getenv("HACX_ACTIVE_PROVIDER")
        if saved_provider and saved_provider in Config.PROVIDERS:
            Config.ACTIVE_PROVIDER = saved_provider
            
        saved_model = os.getenv("HACX_ACTIVE_MODEL")
        if saved_model:
            Config.ACTIVE_MODEL = saved_model

