"""Main CLI module for Eridu."""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """Eridu: Fuzzy matching for entity resolutionthrough representation learning."""
    pass


if __name__ == "__main__":
    cli()
