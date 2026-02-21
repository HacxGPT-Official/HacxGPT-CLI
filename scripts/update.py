import os
import sys
import subprocess
import requests
import zipfile
import io
import shutil

def run_update():
    print("======================================")
    print("    HacxGPT System Updater")
    print("======================================")
    
    VERSION_URL = "https://raw.githubusercontent.com/BlackTechX011/HacxGPT-CLI/main/version.json"
    ARCHIVE_URL = "https://github.com/BlackTechX011/HacxGPT-CLI/archive/refs/heads/main.zip"

    # 1. Check for latest version
    print("[~] Checking for latest version info...")
    try:
        response = requests.get(VERSION_URL, timeout=10)
        if response.status_code == 200:
            remote_version = response.json().get("version", "Unknown")
            print(f"[+] Latest version available: {remote_version}")
        else:
            print("[!] Warning: Could not retrieve version info.")
    except Exception as e:
        print(f"[!] Warning: Version check failed: {e}")

    # 2. Download Latest Archive
    print("[~] Downloading latest system archive...")
    try:
        r = requests.get(ARCHIVE_URL, timeout=30)
        if r.status_code != 200:
            print(f"[!] Download failed with status: {r.status_code}")
            return
        
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # The zip usually contains a folder like HacxGPT-CLI-main
        top_folder = z.namelist()[0].split('/')[0]
        
        print(f"[~] Extracting and updating files...")
        # Extract to a temp directory first to avoid conflicts
        temp_dir = "hacx_update_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        z.extractall(temp_dir)
        source_path = os.path.join(temp_dir, top_folder)
        
        # Copy files over (Surgical update)
        # We only copy the core package and main scripts
        items_to_update = ['hacxgpt', 'scripts', 'setup.py', 'requirements.txt', 'version.json', 'README.md']
        
        for item in items_to_update:
            src = os.path.join(source_path, item)
            dst = os.path.join(os.getcwd(), item)
            if os.path.exists(src):
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        # Cleanup
        shutil.rmtree(temp_dir)
        print("[+] System files synchronized.")
        
    except Exception as e:
        print(f"[!] Error during download/extraction: {e}")
        return

    # 3. Update dependencies
    print("[~] Finalizing installation...")
    try:
        python_exe = sys.executable
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([python_exe, "-m", "pip", "install", "-e", "."])
        print("[+] Dependencies and package re-installed.")
    except Exception as e:
        print(f"[!] Error during dependency update: {e}")
        return

    print("\n======================================")
    print("      Update Complete!")
    print("======================================")
    print("Restart HacxGPT to apply changes.")

if __name__ == "__main__":
    run_update()
