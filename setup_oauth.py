"""
setup_oauth.py
One-time Google OAuth setup script.
Run: python setup_oauth.py
"""

import sys
import yaml
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]


def setup():
    creds_path = CONFIG["credentials_path"]
    token_path = CONFIG["token_path"]

    if not Path(creds_path).exists():
        print(f"\nERROR: '{creds_path}' not found.")
        print("\nTo get credentials.json:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. Enable Google Drive API")
        print("  3. Create OAuth 2.0 Client ID (Desktop App)")
        print("  4. Download JSON → rename to credentials.json")
        print("  5. Place in project folder, then run this script again\n")
        sys.exit(1)

    print(f"Found credentials at: {creds_path}")
    print("Opening browser for Google Drive authorization...")

    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "w") as f:
        f.write(creds.to_json())

    print(f"\nAuthentication successful!")
    print(f"Token saved to: {token_path}")
    print(f"\nDriveAgent is ready.")
    print(f"Run: python agent_loop.py --api     (start REST API)")
    print(f"Run: python agent_loop.py --watch   (watch Drive folder)")
    print(f"Run: python agent_loop.py --file <ID>  (process one file)")


if __name__ == "__main__":
    setup()
