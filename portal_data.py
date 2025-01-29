import os
from requests import get, post
from dataclasses import dataclass
from csv import DictReader, DictWriter
from typing import Dict, Union
from pathlib import Path


def get_offset(file_name: str) -> int:
    with open(file_name, "r") as file:
        reader = DictReader(file)
        last_row = list(reader)[-1]
        return int(last_row["id"])


def export_data(short_name: str, portal_url: str, token: str, file_name: str):
    file_exists = Path(file_name).exists()
    offset = get_offset(file_name) if file_exists else 0
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
            print(f"Requesting {page_size} items at offset: {offset}")
            resp = get(
                f"{portal_url}/api/v2/ai-chatbot/messages/",
                headers=dict(Authorization=f"bearer {token}"),
                params=dict(limit=page_size, offset=offset),
            )
            if resp.status_code != 200:
                break
            json_data = resp.json()
            rows = json_data["results"]
            for row in rows:
                row["short_name"] = short_name
            writer.writerows(rows)
            if not json_data["next"]:
                break
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


def main():
    flow_central_token = os.environ.get("FLOW_CENTRAL_TOKEN")
    if not flow_central_token:
        print("environment variable: FLOW_CENTRAL_TOKEN is not set")
        return
    creds = get_portal_credentials(
        "https://flow.connect.stjohns.edu", flow_central_token
    )
    token = get_portal_token("https://connect.stjohns.edu", creds)
    export_data("stjohns", "https://connect.stjohns.edu", token, "stjohns.csv")


if __name__ == "__main__":
    main()
