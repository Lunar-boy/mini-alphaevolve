from __future__ import annotations

import os
from pathlib import Path

import httpx
import typer
from rich import print as rprint

from minimal_alphaevolve import __version__
from minimal_alphaevolve.config import SaiaSettings
from minimal_alphaevolve.exceptions import MinimalAlphaEvolveError
from minimal_alphaevolve.saia_client import SaiaClient

app = typer.Typer(
    no_args_is_help=True,
    help="Minimal AlphaEvolve-style framework using GWDG SAIA.",
)


@app.command()
def doctor(
    live: bool = typer.Option(
        False, help="Perform a live SAIA POST /models request."
    ),
) -> None:
    '''Check local configuration and optionally verify SAIA access.'''
    key_path = Path.home() / ".config" / "saia" / "api_key"
    env_key_set = bool(os.getenv("SAIA_API_KEY", "").strip())

    rprint(f"[bold]minimal-alphaevolve[/bold] {__version__}")
    key_state = "readable" if os.access(key_path, os.R_OK) else "missing"
    rprint(f"Key file: {key_path} ({key_state})")
    rprint(
        "SAIA_API_KEY environment variable: "
        f"{'set' if env_key_set else 'not set'}"
    )

    settings = SaiaSettings.from_env(require_api_key=live)
    rprint(f"Base URL: {settings.base_url}")
    rprint(f"Model: {settings.model}")

    if not live:
        rprint("[yellow]Live API check skipped. Use --live to call SAIA.[/yellow]")
        return

    try:
        with SaiaClient(settings) as client:
            models = client.list_models()
    except (httpx.HTTPError, MinimalAlphaEvolveError) as exc:
        rprint(f"[red]SAIA check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(f"[green]SAIA access works.[/green] Available models: {len(models)}")
    if settings.model not in models:
        rprint(
            f"[yellow]Configured model {settings.model!r} is not in /models.[/yellow]"
        )


@app.command("models")
def models_command() -> None:
    '''List model identifiers returned by SAIA.'''
    settings = SaiaSettings.from_env()
    try:
        with SaiaClient(settings) as client:
            models = client.list_models()
    except (httpx.HTTPError, MinimalAlphaEvolveError) as exc:
        rprint(f"[red]Unable to list models:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    for model in models:
        marker = " *" if model == settings.model else ""
        typer.echo(f"{model}{marker}")


@app.command()
def smoke(
    prompt: str = typer.Option(
        "Return exactly this JSON object and nothing else: "
        '{"operation":"identity","input":"x0"}',
        help="User prompt for one live chat-completion request.",
    ),
) -> None:
    '''Run one live SAIA chat-completion request.'''
    settings = SaiaSettings.from_env()
    try:
        with SaiaClient(settings) as client:
            completion = client.complete(
                system=(
                    "You are a precise program-synthesis component. "
                    "Follow output-format instructions exactly."
                ),
                user=prompt,
                temperature=0.0,
                max_tokens=256,
                seed=0,
            )
    except (httpx.HTTPError, MinimalAlphaEvolveError) as exc:
        rprint(f"[red]SAIA completion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(f"[bold]Model:[/bold] {completion.model}")
    rprint(completion.content)
