import os
from requests import get, post
from csv import DictReader, DictWriter
from typing import Dict
from pathlib import Path


def get_conversation_csv_path(short_name:str) -> str:
    return f"conversations/{short_name}/{short_name}.csv"


def get_offset(file_name: str) -> int:
    with open(file_name, "r") as file:
        reader = DictReader(file)
        as_list = list(reader)
        if not as_list:
            return 0
        last_row = as_list[-1]
        return int(last_row["id"])


def export_data(short_name: str, portal_api_url: str, token: str, file_name: str):
    csv_file_name = get_conversation_csv_path(short_name)
    Path(csv_file_name).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_file_name).exists()
    offset = get_offset(csv_file_name) if file_exists else 0
    with open(csv_file_name, "a") as file:
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
                "embeddings"
            ],
        )
        if not file_exists:
            writer.writeheader()
        page_size = 10
        num_pages = 1
        while True:
            print(f"  Requesting {page_size} items at offset: {offset}")
            resp = get(
                f"{portal_api_url}/v2/ai-chatbot/messages/",
                headers=dict(Authorization=f"bearer {token}"),
                params=dict(limit=page_size, offset=offset),
            )
            if resp.status_code != 200:
                break
            json_data = resp.json()
            if num_pages % 10 == 0:
                print(f"  Response: {resp.text[:100]}")
            rows = json_data["results"]
            for row in rows:
                row["short_name"] = short_name
                row["embeddings"] = '[]'
                row["content"] = row["content"].replace('\n', ' ') 
            writer.writerows(rows)
            if not json_data["next"]:
                break
            offset += page_size
            num_pages += 1


def get_portal_credentials(short_name: str, flow_url: str, flow_central_token: str) -> dict:
    def find_config(ref_id, body):
        return [x for x in body if x["referenceId"] == ref_id][0]["config"]

    url =f"{flow_url}/repository/sharedConfig?encrypted=true"
    resp = get(
        url,
        headers={
            "flow-central-token": flow_central_token,
            "accept": "application/json"
        }
    )
    body = resp.json()
    cfg = find_config(f"{short_name}_campus", body)
    client_secret = find_config(cfg["client_secret"]["referenceString"], body)
    password = find_config(cfg["password"]["referenceString"], body)

    return {
        "username": cfg["username"],
        "client_id": cfg["client_id"],
        "password": password,
        "client_secret": client_secret,
        "grant_type": "password",
    }


def get_portal_token(portal_api_url: str, creds: dict):
    url=f"{portal_api_url}/oauth/token/"
    resp = post(
        url,
        headers={"Accept": "application/json"},
        json=dict(
            username=creds["username"],
            password=creds["password"],
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            grant_type="password",
        ),
    )
    return resp.json()["access_token"]

def load_servers(file_name) -> Dict[str, str]:
    result = {}
    with open(file_name, "r") as file:
        reader = DictReader(file)
        for row in reader:
            result[row["Env"]] = row["Portal API Url"]
    return result


def export_conversations(short_name, portal_api_url, flow_url, flow_central_token):
    creds = get_portal_credentials(
        short_name, flow_url, flow_central_token
    )
    token = get_portal_token(portal_api_url, creds)
    export_data(short_name, portal_api_url, token, f"{short_name}.csv")


def main():
    flow_central_token = os.environ.get("FLOW_CENTRAL_TOKEN")
    if not flow_central_token:
        print("environment variable: FLOW_CENTRAL_TOKEN is not set")
        return

    servers = load_servers("filtered_envs.csv")
    for short_name, url in servers.items():
        flow_url = url
        if flow_url.endswith("/api"):
            flow_url = flow_url[:-4]
        flow_url = flow_url.replace("https://api.", "https://").replace("https://", "https://flow.")
        print("Exporting conversations for", short_name)
        print(f"  Flow URL: {flow_url}")
        print(f"  Portal URL: {url}")
        export_conversations(short_name, url, flow_url, flow_central_token)
        print()


if __name__ == "__main__":
    main()
