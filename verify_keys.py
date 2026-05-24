import requests
from config import GROQ_API_KEY, GOOGLE_API_KEY


def check_google_key():
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set.")
        return
    print("Checking Gemini/Google API key...")
    url = f"https://generativelanguage.googleapis.com/v1beta2/models?key={GOOGLE_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        print("Google endpoint status:", resp.status_code)
        if resp.ok:
            print("Google API key appears valid.")
            try:
                data = resp.json()
                model_count = len(data.get("models", []))
                print(f"Found {model_count} available model(s).")
            except ValueError:
                print("Response returned non-JSON content.")
        else:
            print("Google key check failed:", resp.text)
    except requests.RequestException as exc:
        print("Google key check error:", exc)


def check_groq_key():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set.")
        return
    print("Checking Groq API key...")
    url = "https://api.groq.com/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print("Groq endpoint status:", resp.status_code)
        if resp.ok:
            print("Groq API key appears valid.")
            try:
                data = resp.json()
                print("Models available:", data)
            except ValueError:
                print("Response returned non-JSON content.")
        else:
            print("Groq key check failed:", resp.text)
    except requests.RequestException as exc:
        print("Groq key check error:", exc)


if __name__ == "__main__":
    print("Verifying configured API keys...")
    print("--------------------------------")
    check_google_key()
    print()
    check_groq_key()
    print("\nIf both keys are valid, your environment is ready for API calls.")
