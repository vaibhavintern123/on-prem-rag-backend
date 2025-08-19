import json
import requests
import google.auth
import google.auth.transport.requests
import os

# ===== 0. Point ADC to your authorized user creds =====
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds_data.json"

# ===== 1. Config =====
ENDPOINT_ID = "43991460227317760"
PROJECT_ID = "947132053690"
LOCATION = "europe-west4"
DEDICATED_DOMAIN = f"{ENDPOINT_ID}.{LOCATION}-{PROJECT_ID}.prediction.vertexai.goog"

def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

messages = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break

    messages.append({"role": "user", "content": user_input})

    payload = {
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": 200
            }
        ]
    }

    access_token = get_access_token()
    url = f"https://{DEDICATED_DOMAIN}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload).json()

    try:
        model_reply = response["predictions"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("Error:", json.dumps(response, indent=2))
        break

    print(f"Gemma: {model_reply.strip()}")

    messages.append({"role": "assistant", "content": model_reply})
