"""
Distillo CLI

Command-line interface for Distillo client operations.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

from distillo import __version__
from distillo.client import DistilloClient

if CLI_AVAILABLE:
    app = typer.Typer(
        name="distillo",
        help="Distillo - Lightweight on/off-policy distillation framework",
        add_completion=False,
    )
    console = Console()


    def version_callback(value: bool):
        """Print version and exit"""
        if value:
            console.print(f"Distillo version {__version__}")
            raise typer.Exit()


    @app.callback()
    def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ):
        """Distillo - Lightweight on/off-policy distillation framework"""
        pass


    @app.command()
    def submit(
        config: str = typer.Argument(..., help="Path to configuration file"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
        wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for job completion"),
        interval: float = typer.Option(5.0, "--interval", "-i", help="Polling interval in seconds"),
        timeout: Optional[float] = typer.Option(
            None, "--timeout", "-t", help="Timeout in seconds"
        ),
    ):
        """
        Submit a job to the Distillo server

        Example:
            distillo submit config.yaml --output result.jsonl
        """
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]Error:[/red] Configuration file not found: {config}")
            raise typer.Exit(1)

        try:
            with console.status("[bold green]Submitting job..."):
                with DistilloClient.from_config(config_path) as client:
                    # Submit job
                    submission = client.submit_job()
                    job_id = submission["job_id"]

                    console.print(f"[green]✓[/green] Job submitted: [bold]{job_id}[/bold]")

                    if wait:
                        # Poll for completion
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            task = progress.add_task("Waiting for completion...", total=None)

                            def update_callback(status):
                                progress.update(
                                    task,
                                    description=f"Status: {status['status']} | Job: {job_id}",
                                )

                            try:
                                status = client.poll_job(
                                    job_id,
                                    interval=interval,
                                    timeout=timeout,
                                    wait_for_completion=True,
                                    callback=update_callback,
                                )
                            except TimeoutError:
                                console.print(
                                    f"[yellow]⚠[/yellow] Job timed out after {timeout} seconds"
                                )
                                raise typer.Exit(1)

                        # Check final status
                        if status["status"] == "completed":
                            console.print(f"[green]✓[/green] Job completed successfully")

                            # Download result if output path specified
                            if output:
                                with console.status("[bold green]Downloading result..."):
                                    client.download_result(job_id, destination=output)
                                console.print(f"[green]✓[/green] Result saved to: {output}")
                            else:
                                console.print(
                                    f"[yellow]ℹ[/yellow] Use 'distillo download {job_id}' to download result"
                                )

                        elif status["status"] == "failed":
                            console.print(f"[red]✗[/red] Job failed")
                            if "message" in status:
                                console.print(f"[red]Error:[/red] {status['message']}")
                            raise typer.Exit(1)

                        elif status["status"] == "cancelled":
                            console.print(f"[yellow]⚠[/yellow] Job was cancelled")
                            raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)


    @app.command()
    def status(
        job_id: str = typer.Argument(..., help="Job ID"),
        watch: bool = typer.Option(False, "--watch", "-w", help="Watch job status"),
        interval: float = typer.Option(5.0, "--interval", "-i", help="Polling interval in seconds"),
    ):
        """
        Check job status

        Example:
            distillo status abc123
            distillo status abc123 --watch
        """
        try:
            # Get config from environment or default
            config_path = Path("config.yaml")
            if not config_path.exists():
                console.print(
                    "[red]Error:[/red] No config.yaml found in current directory. "
                    "Please provide a configuration file."
                )
                raise typer.Exit(1)

            with DistilloClient.from_config(config_path) as client:
                if watch:
                    # Watch mode
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task("Watching job...", total=None)

                        def update_callback(status):
                            progress.update(
                                task, description=f"Status: {status['status']} | Job: {job_id}"
                            )

                        try:
                            status = client.poll_job(
                                job_id,
                                interval=interval,
                                wait_for_completion=True,
                                callback=update_callback,
                            )
                            _print_status(status)
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Stopped watching[/yellow]")
                else:
                    # Single check
                    status = client.get_job_status(job_id)
                    _print_status(status)

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)


    @app.command()
    def download(
        job_id: str = typer.Argument(..., help="Job ID"),
        output: str = typer.Option("result.jsonl", "--output", "-o", help="Output file path"),
    ):
        """
        Download job result

        Example:
            distillo download abc123 --output my_result.jsonl
        """
        try:
            config_path = Path("config.yaml")
            if not config_path.exists():
                console.print(
                    "[red]Error:[/red] No config.yaml found in current directory. "
                    "Please provide a configuration file."
                )
                raise typer.Exit(1)

            with console.status("[bold green]Downloading result..."):
                with DistilloClient.from_config(config_path) as client:
                    client.download_result(job_id, destination=output)

            console.print(f"[green]✓[/green] Result saved to: {output}")

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)


    @app.command()
    def cancel(job_id: str = typer.Argument(..., help="Job ID")):
        """
        Cancel a running job

        Example:
            distillo cancel abc123
        """
        try:
            config_path = Path("config.yaml")
            if not config_path.exists():
                console.print(
                    "[red]Error:[/red] No config.yaml found in current directory. "
                    "Please provide a configuration file."
                )
                raise typer.Exit(1)

            with console.status("[bold green]Cancelling job..."):
                with DistilloClient.from_config(config_path) as client:
                    response = client.cancel_job(job_id)

            console.print(f"[green]✓[/green] Job cancelled: {job_id}")

        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)


    def _print_status(status: dict):
        """Print job status in a formatted table"""
        table = Table(title="Job Status")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Job ID", status.get("job_id", "N/A"))
        table.add_row("Status", status.get("status", "N/A"))

        if "created_at" in status:
            table.add_row("Created", status["created_at"])
        if "started_at" in status:
            table.add_row("Started", status["started_at"])
        if "completed_at" in status:
            table.add_row("Completed", status["completed_at"])

        if "message" in status:
            table.add_row("Message", status["message"])

        if "metrics" in status and status["metrics"]:
            metrics_str = "\n".join(f"{k}: {v}" for k, v in status["metrics"].items())
            table.add_row("Metrics", metrics_str)

        console.print(table)


else:
    # Fallback when typer/rich not available

    def app():
        print(
            "Error: CLI dependencies not installed. Install with: pip install distillo[client]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    if CLI_AVAILABLE:
        app()
    else:
        print(
            "Error: CLI dependencies not installed. Install with: pip install distillo[client]",
            file=sys.stderr,
        )
        sys.exit(1)
