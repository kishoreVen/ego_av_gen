from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logger: logging.Logger = logging.getLogger(__name__)

_SCOPES: List[str] = ["https://www.googleapis.com/auth/drive"]


def build_drive_service(credentials_path: str) -> Any:
    """Build an authenticated Google Drive API service."""
    creds: Credentials = Credentials.from_service_account_file(
        credentials_path, scopes=_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_or_create_folder(
    service: Any,
    folder_name: str,
    parent_id: str,
    folder_cache: Dict[str, str],
) -> str:
    """Find an existing folder or create one under *parent_id*.

    Results are cached in *folder_cache* (key = ``parent_id/folder_name``).
    """
    cache_key: str = f"{parent_id}/{folder_name}"
    if cache_key in folder_cache:
        return folder_cache[cache_key]

    # Search for existing folder
    query: str = (
        f"name='{folder_name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = (
        service.files()
        .list(
            q=query,
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files: List[Dict[str, Any]] = results.get("files", [])

    if files:
        folder_id: str = files[0]["id"]
    else:
        metadata: Dict[str, Any] = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = (
            service.files()
            .create(body=metadata, fields="id", supportsAllDrives=True)
            .execute()
        )
        folder_id = folder["id"]
        logger.info(f"Created GDrive folder: {folder_name} ({folder_id})")

    folder_cache[cache_key] = folder_id
    return folder_id


def delete_existing_file(
    service: Any,
    file_name: str,
    parent_folder_id: str,
) -> None:
    """Delete all files named *file_name* in *parent_folder_id* (prevents duplicates)."""
    query: str = (
        f"name='{file_name}' and '{parent_folder_id}' in parents "
        f"and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    )
    results = (
        service.files()
        .list(
            q=query,
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    for f in results.get("files", []):
        service.files().delete(
            fileId=f["id"], supportsAllDrives=True
        ).execute()
        logger.info(f"Deleted existing GDrive file: {file_name} ({f['id']})")


def upload_file_to_gdrive(
    service: Any,
    local_path: str,
    parent_folder_id: str,
    max_retries: int = 3,
) -> str:
    """Upload a single file to a GDrive folder. Returns the file ID.

    Retries on transient errors (500, 502, 503, 429) with exponential backoff.
    """
    import time

    from googleapiclient.errors import HttpError

    file_name: str = os.path.basename(local_path)
    metadata: Dict[str, Any] = {
        "name": file_name,
        "parents": [parent_folder_id],
    }

    for attempt in range(max_retries + 1):
        try:
            media = MediaFileUpload(local_path, resumable=True)
            uploaded = (
                service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            return uploaded["id"]
        except HttpError as e:
            if e.resp.status in (429, 500, 502, 503) and attempt < max_retries:
                wait: float = 2**attempt
                logger.warning(
                    f"Transient {e.resp.status} uploading {file_name}, "
                    f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise


def _upload_file_thread_safe(
    credentials_path: str,
    local_path: str,
    parent_folder_id: str,
    thread_local: threading.local,
) -> str:
    """Thread-safe file upload — each thread gets its own Drive service."""
    if not hasattr(thread_local, "service"):
        thread_local.service = build_drive_service(credentials_path)
    return upload_file_to_gdrive(thread_local.service, local_path, parent_folder_id)


def upload_directory_to_gdrive(
    local_dir: str,
    folder_id: str,
    credentials_path: str,
    workers: int = 8,
) -> None:
    """Recursively upload *local_dir* contents to a GDrive folder.

    - Creates subdirectories on GDrive to mirror local structure.
    - Follows symlinks so resumed run dirs get uploaded.
    - Uses a ``ThreadPoolExecutor`` for parallel file uploads.
    - Each thread gets its own Drive service (httplib2 is not thread-safe).

    Args:
        local_dir: Local directory to upload.
        folder_id: Target GDrive folder ID.
        credentials_path: Path to service-account JSON key file.
        workers: Number of parallel upload threads.
    """
    service: Any = build_drive_service(credentials_path)
    folder_cache: Dict[str, str] = {}

    # Collect (local_path, gdrive_parent_id) pairs for all files
    upload_tasks: List[tuple[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(local_dir, followlinks=True):
        # Determine the GDrive folder ID for this directory
        rel_path: str = os.path.relpath(dirpath, local_dir)
        if rel_path == ".":
            current_folder_id: str = folder_id
        else:
            current_folder_id = folder_id
            for part in rel_path.split(os.sep):
                current_folder_id = find_or_create_folder(
                    service, part, current_folder_id, folder_cache
                )

        for filename in filenames:
            local_path: str = os.path.join(dirpath, filename)
            upload_tasks.append((local_path, current_folder_id))

    total: int = len(upload_tasks)
    logger.info(f"Uploading {total} files to GDrive folder {folder_id}")

    thread_local: threading.local = threading.local()
    uploaded_count: int = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _upload_file_thread_safe, credentials_path, path, pid, thread_local
            ): path
            for path, pid in upload_tasks
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
                uploaded_count += 1
                if uploaded_count % 20 == 0 or uploaded_count == total:
                    logger.info(f"Uploaded {uploaded_count}/{total} files")
            except Exception:
                logger.exception(f"Failed to upload {path}")

    logger.info(f"GDrive upload complete: {uploaded_count}/{total} files")
