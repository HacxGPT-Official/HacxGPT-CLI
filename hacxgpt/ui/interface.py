
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner
from rich.align import Align
from ..config import Config
from ..core.extractor import CodeExtractor
from .banner import Banner

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.history import FileHistory
from rich.rule import Rule

class UI:
    """Advanced Terminal User Interface using Rich"""
    
    def __init__(self):
        self.console = Console()
        # Create a style for prompt_toolkit that matches our theme
        self.pt_style = Style.from_dict({
            'prompt': 'ansiyellow bold',
        })
        # Initialize session with history
        self.session = PromptSession(history=FileHistory('.hacx_history'))
    
    def clear(self):
        from ..utils.system import clear_screen
        clear_screen()

    def banner(self):
        self.clear()
        Banner.print_banner(self.console)

    def main_menu(self):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Icon", style="bold yellow", justify="right")
        table.add_column("Option", style="bold bright_white")
        
        table.add_row("[1]", "Initialize Uplink [dim](Start Chatting)[/]")
        table.add_row("[2]", "Security Keys [dim](Configure API)[/]")
        table.add_row("[3]", "System Manifesto [dim](About HacxGPT)[/]")
        table.add_row("[4]", "System Update [dim](Check latest version)[/]")
        table.add_row("[5]", "Terminate Session [dim](Exit)[/]")
        
        panel = Panel(
            Align.center(table),
            title="[bold cyan]⚡ SYSTEM INTERFACE ⚡[/bold cyan]",
            border_style="bright_blue",
            padding=(1, 5),
            subtitle="[dim]Select an option to proceed[/]"
        )
        self.console.print(panel)

    def show_msg(self, title: str, content: str, color: str = "white"):
        self.console.print(Panel(content, title=f"[bold]{title}[/]", border_style=color, padding=(1, 2)))

    def get_input(self, label: str = "COMMAND", multiline: bool = False) -> str:
        """
        Get input using prompt_toolkit.
        """
        from prompt_toolkit.key_binding import KeyBindings
        
        kb = KeyBindings()
        
        if multiline:
            @kb.add('enter')
            def _(event):
                event.current_buffer.validate_and_handle()

            @kb.add('escape', 'enter')
            def _(event):
                event.current_buffer.insert_text('\n')
        
        try:
            # Modern prompt style
            self.console.print(f"[bold bright_yellow]◆ {label}[/]")
            
            user_input = self.session.prompt(
                [('class:prompt', ' ╰─> ')],
                style=self.pt_style,
                multiline=multiline,
                key_bindings=kb if multiline else None,
                prompt_continuation=lambda width, line_number, is_soft_wrap: ' ' * (width - 1) + '│'
            )
            return user_input
        except KeyboardInterrupt:
            raise
        except EOFError:
            return "/exit"

    def stream_markdown(self, title: str, content_generator):
        """
        Renders Markdown content in real-time as it streams.
        """
        from rich.console import Group
        
        full_response = ""
        thinking_text = ""
        display_text = ""
        is_thinking = False
        
        # Cleaner header
        self.console.print(Rule(f"[bold bright_cyan]{title}[/bold bright_cyan]", style="bright_blue"))
        
        with Live(
            Spinner("dots", text="Contacting neural network...", style="bright_cyan"),
            console=self.console,
            refresh_per_second=15,
            transient=True
        ) as live:
            
            for chunk in content_generator:
                if not chunk: continue
                full_response += chunk
                
                # Handle thinking tags
                raw_text = full_response
                if "<think>" in raw_text:
                    if "</think>" in raw_text:
                        parts = raw_text.split("</think>")
                        thinking_text = parts[0].replace("<think>", "").strip()
                        display_text = parts[1].strip()
                        is_thinking = False
                    else:
                        thinking_text = raw_text.replace("<think>", "").strip()
                        display_text = ""
                        is_thinking = True
                else:
                    thinking_text = ""
                    display_text = raw_text.strip()
                    is_thinking = False

                ui_elements = []
                
                if thinking_text:
                    ui_elements.append(Panel(
                        thinking_text, 
                        title="[italic dim bright_cyan]Thought Process[/]", 
                        border_style="dim blue", 
                        subtitle="[dim]Analyzing...[/]" if is_thinking else "[dim]Finalizing thought[/]",
                        padding=(0, 1)
                    ))
                
                if display_text:
                    clean_display = display_text.replace("[HacxGPT]:", "").replace("[CODE]:", "").strip()
                    ui_elements.append(Markdown(clean_display, code_theme=Config.CODE_THEME))
                
                if not ui_elements:
                    live.update(Spinner("dots", text="Generating response...", style="bright_cyan"))
                else:
                    live.update(Group(*ui_elements))
            
            if not full_response:
                self.console.print("[bold red]✗ Connection Lost: The neural link went dark.[/]")
            
        # Final render
        final_thinking = ""
        final_response = full_response
        
        if "<think>" in full_response and "</think>" in full_response:
            parts = full_response.split("</think>")
            final_thinking = parts[0].replace("<think>", "").strip()
            final_response = parts[1].strip()
            
            self.console.print(Panel(
                final_thinking, 
                title="[bold bright_cyan]🧠 THOUGHT PROCESS[/]", 
                border_style="bright_blue", 
                style="dim",
                padding=(1, 2)
            ))

        clean_response = final_response.replace("[HacxGPT]:", "").replace("[CODE]:", "").strip()
        md = Markdown(clean_response, code_theme=Config.CODE_THEME)
        self.console.print(md)
        self.console.print(Rule(style="dim bright_blue"))
        
        return full_response

    def handle_code_blocks(self, response_text: str):
        """Handle code block extraction and user actions with Pro UI"""
        code_blocks = CodeExtractor.extract_code_blocks(response_text)
        
        if not code_blocks:
            return
        
        self.console.print(Panel(f"[bold yellow]🔍 Detected {len(code_blocks)} code block(s)[/]", border_style="yellow"))
        
        # Display code blocks info
        table = Table(show_header=True, header_style="bold magenta", border_style="dim white", expand=True)
        table.add_column("#", style="cyan", justify="center", width=4)
        table.add_column("Language", style="green")
        table.add_column("Preview", style="dim white")
        table.add_column("Lines", style="yellow", justify="right")
        
        for idx, (lang, code) in enumerate(code_blocks, 1):
            lines = code.split('\n')
            preview = lines[0].strip()[:50] + "..." if len(lines[0]) > 50 else lines[0].strip()
            table.add_row(str(idx), lang.upper(), preview, str(len(lines)))
        
        self.console.print(table)
        
        # Pro Menu
        menu_text = """
[bold cyan]Options:[/bold cyan]
[bold green][1][/] Save All    [bold green][2][/] Copy All    [bold green][3][/] Save One    [bold green][4][/] Copy One    [bold red][SEMICOLON/Space][/] Skip
"""
        self.console.print(Panel(menu_text.strip(), border_style="blue", title="[bold]Action Menu[/]"))
        self.console.print("[dim]Press the corresponding key...[/]")
        
        from ..utils.system import get_char

        while True:
            char = get_char().lower()
            
            if char == '1':
                self._save_all_blocks(code_blocks)
                break
            elif char == '2':
                self._copy_all_blocks(code_blocks)
                break
            elif char == '3':
                self._save_specific_block_interactive(code_blocks)
                break
            elif char == '4':
                self._copy_specific_block_interactive(code_blocks)
                break
            elif char == ' ' or char == ';':
                self.console.print("[yellow]Skipped.[/]")
                break

    def _save_specific_block_interactive(self, code_blocks):
        self.console.print("[bold cyan]Press the number of the block to save (1-9)...[/]")
        from ..utils.system import get_char
        
        while True:
            char = get_char().lower()
            if not char: continue
            
            if char.isdigit() and 1 <= int(char) <= len(code_blocks):
                idx = int(char) - 1
                lang, code = code_blocks[idx]
                filepath = CodeExtractor.save_code_block(code, lang, idx)
                self.console.print(f"[bold green]✓ Saved Block {char} to: {filepath}[/]")
                break
            # Allow exit on other keys if needed, or just loop

    def _save_all_blocks(self, code_blocks):
        saved_files = []
        for idx, (lang, code) in enumerate(code_blocks):
            filepath = CodeExtractor.save_code_block(code, lang, idx)
            saved_files.append(filepath)
        
        self.console.print(f"\n[bold green]✓ Saved {len(saved_files)} file(s) to: {Config.CODE_OUTPUT_DIR}[/]")

    def _copy_all_blocks(self, code_blocks):
        from ..utils.system import copy_to_clipboard
        
        all_code_parts = []
        for i, (lang, code) in enumerate(code_blocks):
            separator = "=" * 60
            block_header = f"# Block {i+1} - Language: {lang.upper()}"
            all_code_parts.append(f"{separator}\n{block_header}\n{separator}\n\n{code}")
        
        all_code = "\n\n".join(all_code_parts)
        
        if copy_to_clipboard(all_code):
            self.console.print(f"[bold green]✓ All code blocks copied to clipboard! ({len(all_code)} characters)[/]")
        else:
            self.console.print(f"[bold red]✗ Clipboard failed. Please install xclip/xsel (Linux).[/]")

    def _copy_specific_block_interactive(self, code_blocks):
        self.console.print("[bold cyan]Press the number of the block to copy (1-9)...[/]")
        from ..utils.system import get_char, copy_to_clipboard
        
        while True:
            char = get_char().lower()
            if not char: continue
                
            if char.isdigit() and 1 <= int(char) <= len(code_blocks):
                idx = int(char) - 1
                lang, code = code_blocks[idx]
                if copy_to_clipboard(code):
                    self.console.print(f"[bold green]✓ Block {char} copied to clipboard![/]")
                else:
                    self.console.print(f"[bold red]✗ Clipboard failed. Please install xclip/xsel (Linux).[/]")
                break

