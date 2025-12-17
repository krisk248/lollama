"""
Interactive CLI for local LLM chat.

Usage:
    lollama chat --model mistral:7b-instruct-q4_K_M
    lollama chat --system "You are a coding assistant"
    lollama models  # List available models
"""

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None

app = typer.Typer(
    name="lollama",
    help="Local LLM Learning Environment - Interactive CLI",
    add_completion=False,
)
console = Console()


def check_ollama():
    """Check if Ollama is available and running."""
    if ollama is None:
        console.print("[red]Error: ollama package not installed[/]")
        console.print("Run: pip install ollama")
        raise typer.Exit(1)

    try:
        ollama.list()
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to Ollama[/]")
        console.print(f"Details: {e}")
        console.print("\n[yellow]Make sure Ollama is running:[/]")
        console.print("  ollama serve")
        raise typer.Exit(1)


@app.command()
def chat(
    model: str = typer.Option(
        "mistral:7b-instruct-q4_K_M",
        "--model", "-m",
        help="Model to use for chat"
    ),
    system: str = typer.Option(
        "You are a helpful assistant.",
        "--system", "-s",
        help="System prompt"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="Sampling temperature (0.0-1.0)"
    ),
):
    """Start an interactive chat session with a local LLM."""
    check_ollama()

    messages = [{"role": "system", "content": system}]

    console.print(Panel(
        f"[bold green]Chatting with {model}[/]\n"
        f"[dim]Temperature: {temperature}[/]\n"
        f"[dim]System: {system[:50]}...[/]" if len(system) > 50 else f"[dim]System: {system}[/]",
        title="LLM Chat Session",
        border_style="green"
    ))
    console.print("[dim]Type 'exit' or 'quit' to end the session[/]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended[/]")
            break

        if user_input.lower() in ('exit', 'quit', '/exit', '/quit'):
            console.print("[dim]Session ended[/]")
            break

        if not user_input.strip():
            continue

        # Handle special commands
        if user_input.startswith('/'):
            handle_command(user_input, model)
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            with console.status("[bold green]Thinking...[/]"):
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        'temperature': temperature,
                    }
                )

            assistant_message = response['message']['content']
            messages.append({"role": "assistant", "content": assistant_message})

            # Display response
            console.print(f"\n[bold green]Assistant:[/]")
            console.print(Markdown(assistant_message))

            # Show token counts
            prompt_tokens = response.get('prompt_eval_count', 0)
            response_tokens = response.get('eval_count', 0)
            console.print(f"\n[dim]Tokens: {prompt_tokens} prompt + {response_tokens} response[/]\n")

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


def handle_command(command: str, model: str):
    """Handle special commands."""
    cmd = command.lower().strip()

    if cmd == '/help':
        console.print(Panel(
            "[bold]Available Commands:[/]\n\n"
            "/help - Show this help\n"
            "/clear - Clear conversation history\n"
            "/model - Show current model\n"
            "/models - List available models\n"
            "/exit - End the session",
            title="Help",
            border_style="blue"
        ))
    elif cmd == '/model':
        console.print(f"[green]Current model: {model}[/]")
    elif cmd == '/models':
        models_cmd()
    elif cmd == '/clear':
        console.print("[yellow]Conversation history cleared (restart to apply)[/]")
    else:
        console.print(f"[yellow]Unknown command: {command}[/]")
        console.print("[dim]Type /help for available commands[/]")


@app.command()
def models():
    """List all available Ollama models."""
    check_ollama()

    try:
        result = ollama.list()
        models_list = result.get('models', [])

        if not models_list:
            console.print("[yellow]No models installed.[/]")
            console.print("\nInstall a model with:")
            console.print("  ollama pull mistral:7b-instruct-q4_K_M")
            return

        table = Table(title="Installed Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="dim")
        table.add_column("Quantization", style="yellow")

        for model in models_list:
            name = model.get('name', 'Unknown')
            size_bytes = model.get('size', 0)
            size_gb = size_bytes / (1024 ** 3)
            modified = model.get('modified_at', 'Unknown')[:10]
            details = model.get('details', {})
            quant = details.get('quantization_level', 'Unknown')

            table.add_row(name, f"{size_gb:.2f} GB", modified, quant)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/]")


# Alias for models command
models_cmd = models


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt for generation"),
    model: str = typer.Option(
        "mistral:7b-instruct-q4_K_M",
        "--model", "-m",
        help="Model to use"
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens", "-n",
        help="Maximum tokens to generate"
    ),
):
    """Generate text completion (non-interactive)."""
    check_ollama()

    try:
        with console.status("[bold green]Generating...[/]"):
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={'num_predict': max_tokens}
            )

        console.print(Markdown(response['response']))

        # Show stats
        tokens = response.get('eval_count', 0)
        duration = response.get('total_duration', 0) / 1e9  # ns to s
        tps = tokens / duration if duration > 0 else 0
        console.print(f"\n[dim]{tokens} tokens in {duration:.2f}s ({tps:.1f} tok/s)[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model to download (e.g., mistral:7b)"),
):
    """Download a model from Ollama registry."""
    check_ollama()

    console.print(f"[bold]Downloading {model}...[/]")

    try:
        for progress in ollama.pull(model, stream=True):
            status = progress.get('status', '')
            completed = progress.get('completed', 0)
            total = progress.get('total', 0)

            if total > 0:
                pct = (completed / total) * 100
                console.print(f"\r[green]{status}: {pct:.1f}%[/]", end="")
            else:
                console.print(f"\r[green]{status}[/]", end="")

        console.print(f"\n[bold green]Successfully downloaded {model}[/]")

    except Exception as e:
        console.print(f"\n[red]Error downloading model: {e}[/]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system and Ollama information."""
    check_ollama()

    import psutil
    import platform

    # System info
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("OS", platform.system())
    table.add_row("Python", platform.python_version())
    table.add_row("CPU Cores", str(psutil.cpu_count(logical=False)))
    table.add_row("Logical CPUs", str(psutil.cpu_count(logical=True)))

    mem = psutil.virtual_memory()
    table.add_row("Total RAM", f"{mem.total / (1024**3):.1f} GB")
    table.add_row("Available RAM", f"{mem.available / (1024**3):.1f} GB")
    table.add_row("RAM Usage", f"{mem.percent}%")

    console.print(table)

    # Model count
    try:
        result = ollama.list()
        model_count = len(result.get('models', []))
        console.print(f"\n[bold]Ollama Models Installed:[/] {model_count}")
    except Exception:
        console.print("\n[yellow]Could not get Ollama model count[/]")


if __name__ == "__main__":
    app()
