from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / ".env.example"
DEST = ROOT / ".env"

if __name__ == "__main__":
    if not EXAMPLE.exists():
        raise FileNotFoundError(".env.example not found. Create it first.")

    if DEST.exists():
        print(".env already exists. Open it and fill in your API keys.")
        print("If you want to overwrite it, delete .env and rerun this script.")
    else:
        shutil.copy(EXAMPLE, DEST)
        print("Copied .env.example to .env.")
        print("Open .env and paste your API keys for GROQ_API_KEY and GOOGLE_API_KEY.")
        print("Then run 'python config.py' to verify your setup.")
