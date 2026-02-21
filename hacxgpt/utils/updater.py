
import os
import sys
import subprocess
import requests
from .. import __version__

class Updater:
    """Handles checking for and applying updates from GitHub."""
    
    REPO_URL = "https://api.github.com/repos/BlackTechX011/HacxGPT-CLI/releases/latest"
    GITHUB_RAW_VERSION = "https://raw.githubusercontent.com/BlackTechX011/HacxGPT-CLI/main/version.json"

    @staticmethod
    def get_remote_version():
        """Fetches the version string from version.json on GitHub."""
        try:
            response = requests.get(Updater.GITHUB_RAW_VERSION, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except Exception:
            return None
        return None

    @staticmethod
    def check_for_updates():
        """Compares local version with remote version."""
        remote_version = Updater.get_remote_version()
        if not remote_version:
            return False, "Could not reach update server."
        
        # Simple version comparison
        if remote_version > __version__:
            return True, remote_version
        return False, __version__

    @staticmethod
    def update():
        """Executes the external update script."""
        print("[~] Initiating System Update...")
        
        try:
            # We execute scripts/update.py using the current python interpreter
            update_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "update.py")
            if not os.path.exists(update_script):
                # Fallback check relative to CWD
                update_script = os.path.join(os.getcwd(), "scripts", "update.py")
            
            if not os.path.exists(update_script):
                return False, "Update script not found in scripts/update.py."

            subprocess.check_call([sys.executable, update_script])
            return True, "Update successful. Please restart HacxGPT."
        except subprocess.CalledProcessError as e:
            return False, f"Update failed during execution: {e}"
        except Exception as e:
            return False, f"An unexpected error occurred: {e}"
