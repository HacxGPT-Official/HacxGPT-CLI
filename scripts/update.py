
import os
import sys
import subprocess

def run_update():
    print("======================================")
    print("    HacxGPT System Updater")
    print("======================================")
    
    # 1. Check for Git
    try:
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL)
    except:
        print("[!] Error: Git is not installed. Manual update required.")
        return

    # 2. Check if .git exists
    if not os.path.exists(".git"):
        print("[!] Error: Not a git repository. Please clone from GitHub.")
        return

    # 3. Pull from GitHub
    print("[~] Fetching latest changes from main branch...")
    try:
        subprocess.check_call(["git", "pull", "origin", "main"])
        print("[+] Codebase synchronized.")
    except Exception as e:
        print(f"[!] Error during git pull: {e}")
        return

    # 4. Update dependencies
    print("[~] Updating dependencies...")
    try:
        # Detect venv
        python_exe = sys.executable
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([python_exe, "-m", "pip", "install", "-e", "."])
        print("[+] Dependencies updated.")
    except Exception as e:
        print(f"[!] Error during dependency update: {e}")
        return

    print("
======================================")
    print("      Update Complete!")
    print("======================================")
    print("Restart HacxGPT to apply changes.")

if __name__ == "__main__":
    run_update()
