"""
drive_connector.py
Google Drive API v3 - OAuth2 auth, list/download/upload files.
Matches architecture spec: drive_connector.py
"""

import io
import os
from pathlib import Path
from typing import Optional
import yaml

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Load config
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

EXPORT_MAP = {
    "application/vnd.google-apps.document":     ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet":  ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "application/vnd.google-apps.presentation": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    "application/vnd.google-apps.drawing":      ("image/png", ".png"),
}


class DriveConnector:
    """Google Drive API v3 connector with OAuth2."""

    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        token_path = CONFIG["token_path"]
        creds_path = CONFIG["credentials_path"]

        if Path(token_path).exists():
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing Google credentials...")
                creds.refresh(Request())
            else:
                if not Path(creds_path).exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at '{creds_path}'.\n"
                        "Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(token_path, "w") as f:
                f.write(creds.to_json())

        print("Google Drive authenticated.")
        return build("drive", "v3", credentials=creds)

    def get_file_metadata(self, file_id: str) -> dict:
        return self.service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size, modifiedTime"
        ).execute()

    def list_files(self, folder_id: Optional[str] = None, page_size: int = 20) -> list:
        folder_id = folder_id or CONFIG.get("drive_folder_id", "root")
        query = f"'{folder_id}' in parents and trashed=false"
        result = self.service.files().list(
            q=query,
            pageSize=page_size,
            fields="files(id, name, mimeType, size, modifiedTime)"
        ).execute()
        return result.get("files", [])

    def download_file(self, file_id: str, dest_path: str) -> str:
        meta = self.get_file_metadata(file_id)
        mime_type = meta["mimeType"]
        name = meta["name"]

        if mime_type in EXPORT_MAP:
            export_mime, ext = EXPORT_MAP[mime_type]
            dest_path = str(Path(dest_path).with_suffix(ext))
            print(f"Exporting Google file '{name}' as {ext}")
            request = self.service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            print(f"Downloading '{name}'")
            request = self.service.files().get_media(fileId=file_id)

        os.makedirs(Path(dest_path).parent, exist_ok=True)
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        print(f"Downloaded to: {dest_path}")
        return dest_path

    def upload_file(self, local_path: str, file_id: Optional[str] = None,
                    folder_id: Optional[str] = None) -> str:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(local_path)
        mime_type = mime_type or "application/octet-stream"
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

        if file_id:
            result = self.service.files().update(
                fileId=file_id, media_body=media, fields="id, name"
            ).execute()
        else:
            file_meta = {"name": Path(local_path).name}
            if folder_id:
                file_meta["parents"] = [folder_id]
            result = self.service.files().create(
                body=file_meta, media_body=media, fields="id, name"
            ).execute()

        print(f"Uploaded: {result['name']} (id={result['id']})")
        return result["id"]
