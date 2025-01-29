from requests import get, post
from dataclasses import dataclass
from csv import DictReader, DictWriter
from typing import Dict, Union
from pathlib import Path

CustomerName = str


@dataclass
class ExportStatus:
    short_name: str
    last_offset: int


def load_export_status(file_name: str) -> Dict[CustomerName, ExportStatus]:
    with open(file_name, "r") as file:
        reader = DictReader(file)
        return {
            row["short_name"]: ExportStatus(row["short_name"], int(row["last_offset"]))
            for row in reader
        }


export_status = load_export_status("export_status.csv")


def export_data(
    short_name: str, portal_url: str, token: str, file_name: str, offset: int = 0
):
    file_exists = Path(file_name).exists()
    with open(file_name, "a") as file:
        writer = DictWriter(
            file,
            fieldnames=[
                "short_name",
                "id",
                "conversation",
                "flow_id",
                "timestamp",
                "content",
                "answer_type",
                "failed",
                "author",
            ],
        )
        if not file_exists:
            writer.writeheader()
        page_size = 10
        while True:
            print(f"Requesting page: {offset}")
            resp = get(
                f"{portal_url}/api/v2/ai-chatbot/messages/",
                headers=dict(Authorization=f"bearer {token}"),
                params=dict(limit=page_size, offset=offset),
            )
            if resp.status_code != 200:
                break
            rows = resp.json()["results"]
            for row in rows:
                row["short_name"] = short_name
            writer.writerows(rows)
            offset += page_size


def get_portal_credentials(flow_url: str, flow_central_token: str) -> dict:
    def find_config(ref_id, body):
        return [x for x in body if x["referenceId"] == ref_id][0]["config"]

    resp = get(
        f"{flow_url}/repository/sharedConfig?encrypted=true",
        headers={"flow-central-token": flow_central_token},
    )
    body = resp.json()
    cfg = find_config("stjohns_campus", body)
    client_secret = find_config(cfg["client_secret"]["referenceString"], body)
    password = find_config(cfg["password"]["referenceString"], body)

    return {
        "username": cfg["username"],
        "client_id": cfg["client_id"],
        "password": password,
        "client_secret": client_secret,
        "grant_type": "password",
    }


def get_portal_token(portal_url: str, creds: dict):
    resp = post(
        f"{portal_url}/api/oauth/token/",
        json=dict(
            username=creds["username"],
            password=creds["password"],
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            grant_type="password",
        ),
    )
    return resp.json()["access_token"]


creds = get_portal_credentials(
    "https://flow.connect.stjohns.edu", ""
)
token = get_portal_token("https://connect.stjohns.edu", creds)
export_data("stjohns", "https://connect.stjohns.edu", token, "stjohns.csv", 0)
