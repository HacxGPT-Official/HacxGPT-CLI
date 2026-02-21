
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
        """Performs a git pull and re-installs dependencies."""
        print("[~] Initiating System Update...")
        
        # 1. Check if it's a git repo
        if not os.path.exists(".git"):
            return False, "Not a git repository. Manual update required."

        try:
            # 2. Pull latest changes
            print("[~] Fetching latest code from uplink...")
            subprocess.check_call(["git", "pull", "origin", "main"])
            
            # 3. Update dependencies
            print("[~] Synchronizing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            
            return True, "Update successful. Please restart HacxGPT."
        except subprocess.CalledProcessError as e:
            return False, f"Update failed during execution: {e}"
        except Exception as e:
            return False, f"An unexpected error occurred: {e}"
