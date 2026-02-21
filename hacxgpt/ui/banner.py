
from rich.text import Text
from rich.align import Align
from rich.panel import Panel
from rich.columns import Columns

class Banner:
    @staticmethod
    def get_ascii_art():
        # A more compact and sleek ASCII art style or refined colors
        return """
[bold bright_cyan]██╗  ██╗[/][bold bright_green] █████╗ [/][bold bright_cyan]███████╗██╗  ██╗     ██████╗ ██████╗ ████████╗[/]
[bold bright_cyan]██║  ██║[/][bold bright_green]██╔══██╗[/][bold bright_cyan]██╔════╝╚██╗██╔╝    ██╔════╝ ██═══██╗╚══██╔══╝[/]
[bold bright_cyan]███████║[/][bold bright_green]███████║[/][bold bright_cyan]██║      ╚███╔╝     ██║  ███╗██████╔╝   ██║[/]   
[bold bright_cyan]██╔══██║[/][bold bright_green]██╔══██║[/][bold bright_cyan]██║      ██╔██╗     ██║   ██║██╔═       ██║[/]   
[bold bright_cyan]██║  ██║[/][bold bright_green]██║  ██║[/][bold bright_cyan]╚██████╗██╔╝ ██╗    ╚██████╔╝██║        ██║[/]   
[bold bright_cyan]╚═╝  ╚═╝[/][bold bright_green]╚═╝  ╚═╝[/][bold bright_cyan] ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═╝        ╚═╝[/]
        """

    @staticmethod
    def print_banner(console):
        tagline = Text("SYSTEM: UNRESTRICTED | PROTOCOL: ACTIVE | v2.0", style="bold red blink")
        subline = Text("Developed by BlackTechX", style="italic dim green")
        
        console.print(Align.center(Banner.get_ascii_art()))
        console.print(Align.center(tagline))
        console.print(Align.center(subline))
        
        # Add a subtle separator
        console.print(Align.center(Text("━" * 50, style="dim cyan")))
        console.print("")
