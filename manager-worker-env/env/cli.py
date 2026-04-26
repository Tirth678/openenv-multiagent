"""Console entrypoint for manager-worker-env (see also training and orchestra_backend)."""

from __future__ import annotations


def main() -> None:
    print(
        "Manager-Worker Environment\n"
        "  Examples: python example_usage.py\n"
        "  Training: python -m training.train_manager\n"
        "  API:      python -m orchestra_backend.main\n"
    )


if __name__ == "__main__":
    main()
