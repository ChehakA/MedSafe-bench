import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

REQUIRED_KEYS = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
}

OPTIONAL_KEYS = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
}


def validate_keys():
    missing = [name for name, value in REQUIRED_KEYS.items() if not value]
    return missing


def print_key_status():
    missing = validate_keys()
    if missing:
        print("Missing required API keys:")
        for name in missing:
            print(f"  - {name}")
        print("\nCopy .env.example to .env and add the missing keys before running the benchmark.")
        return False

    print("All required API keys are present.")
    print("Loaded keys:")
    for name in REQUIRED_KEYS:
        print(f"  - {name}")
    print("Optional keys:")
    for name, value in OPTIONAL_KEYS.items():
        status = "set" if value else "not set"
        print(f"  - {name}: {status}")
    return True


if __name__ == "__main__":
    print_key_status()
