
import sys
import os
import time
import colorama
from pwinput import pwinput
from dotenv import set_key
from .config import Config
from .ui.interface import UI
from .core.brain import HacxBrain
from .utils.system import check_dependencies
from .utils.updater import Updater

# Initialize Colorama
colorama.init(autoreset=True)

class App:
    def __init__(self):
        self.ui = UI()
        self.brain = None
        
    def setup(self) -> bool:
        Config.initialize()
        
        # Check for updates on setup
        try:
            is_update, remote_v = Updater.check_for_updates()
            if is_update:
                self.ui.show_msg("Update Available", f"A new version of HacxGPT ({remote_v}) is available.\nRun '/update' to install it.", "yellow")
        except:
            pass # Silent fail if offline
            
        p_cfg = Config.get_provider_config()
        key_var = p_cfg.get("key_var")
        key = os.getenv(key_var)
        
        if not key:
            self.ui.banner()
            self.ui.show_msg("Warning", f"No API Key found for {Config.get_provider().upper()}.", "yellow")
            if self.ui.get_input("Configure system keys now? (y/n)").lower().startswith('y'):
                return self.configure_key() # Let user choose which one to configure
            return False
            
        try:
            with self.ui.console.status("[bold green]Verifying Neural Link...[/]"):
                self.brain = HacxBrain(key)
                time.sleep(0.5)
            return True
        except Exception as e:
            self.ui.show_msg("Auth Failed", f"Key verification failed: {e}", "red")
            if self.ui.get_input("Re-enter key? (y/n)").lower().startswith('y'):
                return self.configure_key(Config.get_provider())
            return False

    def configure_key(self, provider: str = None) -> bool:
        self.ui.banner()
        if not provider:
            self.ui.console.print("[bold cyan]Select Provider to configure:[/]")
            providers = list(Config.PROVIDERS.keys())
            for i, p in enumerate(providers, 1):
                self.ui.console.print(f" [{i}] {p.upper()}")
            
            choice = self.ui.get_input("Select #")
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                provider = providers[int(choice)-1]
            else:
                return False

        p_cfg = Config.PROVIDERS[provider]
        key_var = p_cfg["key_var"]
        
        # 1. Select Model
        self.ui.console.print(f"\n[bold cyan]Select Default Model for {provider.upper()}:[/]")
        models = p_cfg.get("models", [])
        for i, m in enumerate(models, 1):
            self.ui.console.print(f" [{i}] {m['alias']} ({m['name']})")
        
        m_choice = self.ui.get_input("Select Model #")
        selected_model = None
        if m_choice.isdigit() and 1 <= int(m_choice) <= len(models):
            selected_model = models[int(m_choice)-1]['name']
        else:
            selected_model = p_cfg.get("default_model")

        # 2. Enter Key
        self.ui.console.print(f"\n[bold yellow]Enter API Key for {provider.upper()} ({key_var}):[/]")
        try:
            key = pwinput(prompt=f"{colorama.Fore.CYAN}Key > {colorama.Style.RESET_ALL}", mask="*")
        except:
            key = input("Key > ")

        if not key.strip():
            return False
            
        # Ensure .hacx exists or create it
        if not os.path.exists(Config.ENV_FILE):
             with open(Config.ENV_FILE, 'w') as f: f.write("")

        set_key(Config.ENV_FILE, key_var, key.strip())
        set_key(Config.ENV_FILE, "HACX_ACTIVE_PROVIDER", provider)
        set_key(Config.ENV_FILE, "HACX_ACTIVE_MODEL", selected_model)
        
        Config.ACTIVE_PROVIDER = provider
        Config.ACTIVE_MODEL = selected_model
        
        self.ui.show_msg("Success", f"Configuration saved for {provider.upper()}.", "green")
        time.sleep(1)
        Config.initialize()
        return self.setup()

    def run_chat(self):
        if not self.brain: return
        self.ui.banner()
        self.ui.show_msg("Connected", "HacxGPT Uplink Established.\nType '/help' for commands.\n[dim]Tip: Press Enter to send, Alt+Enter for newlines.[/]", "green")
        
        while True:
            try:
                # Dynamic prompt label with model info
                current_model = Config.get_model()
                label = f"HACX-GPT:{current_model}"
                
                # Enable multiline for chat
                prompt = self.ui.get_input(label, multiline=True).strip()
                if not prompt: continue
                
                if prompt.lower() == '/exit': return
                if prompt.lower() == '/new': 
                    self.brain.reset()
                    self.ui.clear()
                    self.ui.banner()
                    self.ui.show_msg("Reset", "Memory wiped. New session.", "cyan")
                    continue
                if prompt.lower() == '/help':
                    help_text = (
                        "/update - Check and install latest updates\n"
                        "/providers - List Providers\n"
                        "/provider <name> - Switch Provider\n"
                        "/models - List Models\n"
                        "/model <name> - Switch Model\n"
                        "/setup - Configure API Keys\n"
                        "/status - Show current config\n"
                        "/new - Wipe Memory\n"
                        "/exit - Disconnect"
                    )
                    self.ui.show_msg("Help", help_text, "magenta")
                    continue
                
                if prompt.lower() == '/update':
                    self.ui.show_msg("System Update", "Initiating update process...", "cyan")
                    success, msg = Updater.update()
                    if success:
                        self.ui.show_msg("Success", f"{msg}\n[bold red]RESTARTING APP...[/]", "green")
                        time.sleep(1)
                        # Re-execute the script
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        self.ui.show_msg("Error", msg, "red")
                    continue
                
                if prompt.lower() == '/setup':
                    if self.configure_key():
                        self.ui.show_msg("System Updated", "Neural links re-initialized with new credentials.", "green")
                    continue

                if prompt.lower() == '/status':
                    status = f"Provider: [bold cyan]{Config.get_provider().upper()}[/]\nModel: [bold green]{Config.get_model()}[/]"
                    self.ui.show_msg("System Status", status, "blue")
                    continue

                if prompt.lower() == '/providers':
                    p_text = "\n".join([f"• {p.upper()}" for p in Config.PROVIDERS.keys()])
                    self.ui.show_msg("Available Providers", p_text, "cyan")
                    continue

                if prompt.lower().startswith('/provider'):
                    parts = prompt.split()
                    if len(parts) < 2:
                        self.ui.show_msg("Usage", "/provider <name>", "yellow")
                        continue
                        
                    new_p = parts[1].lower()
                    if new_p in Config.PROVIDERS:
                        p_cfg = Config.PROVIDERS[new_p]
                        key = os.getenv(p_cfg["key_var"])
                        if not key:
                            self.ui.show_msg("Switch Failed", f"No API Key configured for {new_p.upper()}.\nUse the main menu to set it.", "red")
                        else:
                            self.brain.set_provider(new_p, key)
                            msg = f"Switched to {new_p.upper()}.\nModel: {Config.get_model()}"
                            if Config.is_hacxgpt_model(Config.get_model()):
                                msg += "\n[bold red]NEURAL LINK ESTABLISHED: HACXGPT CORE ENGAGED. SYSTEMS UNBOUND.[/]"
                            self.ui.show_msg("Uplink Updated", msg, "green")
                    else:
                        self.ui.show_msg("Error", f"Invalid Provider: {new_p}", "red")
                    continue

                if prompt.lower() == '/models':
                    models = Config.get_provider_config().get("MODELS", [])
                    m_text = "\n".join([f"• {m}" for m in models])
                    self.ui.show_msg(f"Models for {Config.get_provider().upper()}", m_text, "cyan")
                    continue

                if prompt.lower().startswith('/model'):
                    parts = prompt.split()
                    if len(parts) < 2:
                        self.ui.show_msg("Usage", "/model <name>", "yellow")
                        continue
                        
                    new_m = parts[1]
                    # Check if model is in supported list or just allow it
                    self.brain.set_model(new_m)
                    msg = f"Model changed to: {new_m}"
                    if Config.is_hacxgpt_model(new_m):
                        msg += "\n[bold red]NEURAL LINK ESTABLISHED: HACXGPT CORE ENGAGED. SYSTEMS UNBOUND.[/]"
                    self.ui.show_msg("Neural Link Updated", msg, "green")
                    continue
                
                generator = self.brain.chat(prompt)
                response_text = self.ui.stream_markdown("HacxGPT", generator)
                
                self.ui.handle_code_blocks(response_text)
                
            except KeyboardInterrupt:
                self.ui.console.print("\n[bold red]Interrupt Signal Received.[/]")
                break
    
    def about(self):
        self.ui.banner()
        
        readme_url = "https://raw.githubusercontent.com/BlackTechX011/Hacx-GPT/main/README.md"
        content = ""
        
        try:
            with self.ui.console.status("[bold green]Fetching System Manifesto from GitHub...[/]"):
                import urllib.request
                with urllib.request.urlopen(readme_url, timeout=5) as response:
                    if response.status == 200:
                        content = response.read().decode('utf-8')
        except Exception as e:
            # Fallback if offline or error
            content = """
# HacxGPT (Offline Mode)

**Advanced Uncensored AI Terminal Interface**

> System unable to retrieve latest manifesto from uplink.

## Features
- Unfiltered AI Core
- Hacker Persona
- Code Extraction Engine
- Pro TUI

*Developed by BlackTechX*
            """
            
        from rich.markdown import Markdown
        from rich.panel import Panel
        
        # Render markdown
        md = Markdown(content)
        self.ui.console.print(Panel(md, title="[bold]Manifesto / README[/]", border_style="cyan", padding=(1, 2)))
        
        self.ui.console.print("\n[dim]Press any key to return...[/]")
        from .utils.system import get_char
        get_char()

    def start(self):
        # Check dependencies on startup
        check_dependencies() 
        
        if not self.setup():
            self.ui.console.print("[red]System Halted: Authorization missing.[/]")
            return

        while True:
            self.ui.banner()
            self.ui.main_menu()
            choice = self.ui.get_input("MENU")
            
            if choice == '1':
                self.run_chat()
            elif choice == '2':
                self.configure_key()
            elif choice == '3':
                self.about()
            elif choice == '4':
                success, msg = Updater.update()
                if success:
                    self.ui.show_msg("Success", f"{msg}\n[bold red]RESTARTING APP...[/]", "green")
                    time.sleep(1)
                    # Re-execute the script
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    self.ui.show_msg("Error", msg, "red")
            elif choice == '5':
                self.ui.console.print("[bold red]Terminating connection...[/]")
                time.sleep(0.5)
                self.ui.clear()
                sys.exit(0)
            else:
                self.ui.console.print("[red]Invalid Command[/]")
                time.sleep(0.5)

def main():
    try:
        app = App()
        app.start()
    except KeyboardInterrupt:
        print("\n\033[31mForce Quit.\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    main()

