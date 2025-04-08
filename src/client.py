import argparse
import sys
import termios
import time
import tty
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


def read_multiline_input_unix(console: Console) -> Optional[str]:
    prompt_lines: List[str] = []
    line: str = ""

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        console.print(
            "[bold yellow]Enter your prompt (press Ctrl+Enter when finished):[/bold yellow]"
        )

        while True:
            char: str = sys.stdin.read(1)

            if char == "\r" or char == "\n":
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                line += "\n"

                if prompt_lines and prompt_lines[-1] and ord(prompt_lines[-1][-1]) < 32:
                    prompt_lines[-1] = prompt_lines[-1][:-1]
                    break

                prompt_lines.append(line)
                line = ""
            elif char in ("\x7f", "\x08"):
                if line:
                    line = line[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif char == "\x03":
                raise KeyboardInterrupt
            else:
                line += char
                sys.stdout.write(char)
                sys.stdout.flush()

    except KeyboardInterrupt:
        console.print("\n[bold red]Input cancelled by user[/bold red]")
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return "".join(prompt_lines).rstrip()


def read_multiline_input_windows(console: Console) -> Optional[str]:
    console.print(
        "[bold yellow]Enter your prompt (press Enter followed by Ctrl+Z and Enter again when finished):[/bold yellow]"
    )
    prompt_lines: List[str] = []
    try:
        for line in sys.stdin:
            prompt_lines.append(line)
        return "".join(prompt_lines).rstrip()
    except KeyboardInterrupt:
        console.print("\n[bold red]Input cancelled by user[/bold red]")
        return None


def get_prompt_input(console: Console, prompt_arg: Optional[str]) -> Optional[str]:
    if prompt_arg:
        return prompt_arg

    if sys.platform in ["linux", "darwin"]:
        prompt = read_multiline_input_unix(console)
    else:
        prompt = read_multiline_input_windows(console)

    if prompt is None:
        return None

    if not prompt:
        console.print("[bold red]Error: Empty prompt provided[/bold red]")
        return None

    return prompt


def fetch_model_info(console: Console, url: str) -> bool:
    try:
        with console.status("[bold green]Fetching model information..."):
            response = requests.get(f"{url}/")

        model_info: Dict[str, Any] = response.json()
        table = Table(
            title="Model Information", show_header=True, header_style="bold magenta"
        )

        for key in model_info.keys():
            table.add_column(key.replace("_", " ").title(), style="cyan")

        table.add_row(*[str(val) for val in model_info.values()])
        console.print(table)
        console.print()
        return True
    except Exception as e:
        console.print(f"[bold red]Error getting model info:[/bold red] {e}")
        return False


def generate_text(
    console: Console,
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    steps: int,
    alg: str,
    alg_temp: float,
) -> bool:
    try:
        prompt_panel = Panel(prompt, title="Prompt", border_style="green", expand=False)
        console.print(prompt_panel)

        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "steps": steps,
            "alg": alg,
            "alg_temp": alg_temp,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Generating text..."),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=None)
            start_time = time.time()
            response = requests.post(f"{url}/generate", json=payload)
            response_json: Dict[str, Any] = response.json()
            end_time = time.time()
            progress.update(task, completed=True)

        display_results(console, response_json, end_time - start_time)
        return True
    except Exception as e:
        console.print(f"[bold red]Error generating text:[/bold red] {e}")
        return False


def display_results(
    console: Console, response_json: Dict[str, Any], generation_time: float
) -> None:
    generated_text = Text(response_json["generated_text"])
    output_panel = Panel(
        generated_text,
        title=f"Generated Text (took {generation_time:.2f} seconds)",
        border_style="blue",
        expand=False,
    )
    console.print(output_panel)

    model_info_text = f"Model: [bold]{response_json['model_name']}[/bold]"
    if "gpu_id" in response_json:
        model_info_text += f", GPU: [bold]{response_json['gpu_id']}[/bold]"
    else:
        model_info_text += ", GPU: [italic]N/A[/italic]"

    console.print(model_info_text)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dream Model Server Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the server")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=512,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="entropy",
        help="Algorithm to use (e.g., 'entropy')",
    )
    parser.add_argument(
        "--alg_temp",
        type=float,
        default=0.0,
        help="Algorithm temperature parameter",
    )
    return parser.parse_args()


def main() -> None:
    console = Console()
    args = parse_arguments()
    url = f"http://{args.host}:{args.port}"

    if not fetch_model_info(console, url):
        return

    prompt = get_prompt_input(console, args.prompt)
    if prompt is None:
        return

    generate_text(
        console,
        url,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.steps,
        args.alg,
        args.alg_temp,
    )


if __name__ == "__main__":
    main()
