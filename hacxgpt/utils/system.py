
import sys
import subprocess
import time
import os

def check_dependencies():
    """Checks and auto-installs missing dependencies."""
    required_packages = [
        ("colorama", "colorama"),
        ("pwinput", "pwinput"),
        ("dotenv", "python-dotenv"),
        ("rich", "rich"),
        ("pyperclip", "pyperclip"),
        ("requests", "requests")
    ]
    
    missing_pip_names = []
    
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_pip_names.append(pip_name)
            
    if missing_pip_names:
        print(f"[\033[93m!\033[0m] Missing dependencies: {', '.join(missing_pip_names)}")
        print("[\033[96m*\033[0m] Installing automatically...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_pip_names])
            print("[\033[92m+\033[0m] Installation complete. Restarting script...")
            time.sleep(1)
            # Re-execute the script
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print("Please manually run: pip install " + " ".join(missing_pip_names))
            sys.exit(1)

def clear_screen():
    """Clears the terminal screen (Cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_char() -> str:
    """
    Reads a single character from stdin (Cross-platform).
    Returns the character as a string.
    """
    if os.name == 'nt':
        import msvcrt
        try:
            return msvcrt.getch().decode('utf-8')
        except UnicodeDecodeError:
            return '\x00' # Return null char in case of weird keypress
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def copy_to_clipboard(text: str) -> bool:
    """
    Copies text to clipboard using pyperclip.
    Handles errors if no clipboard mechanism is found (e.g. on Linux without xclip).
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        # This occurs if pyperclip can't find a backend
        return False

