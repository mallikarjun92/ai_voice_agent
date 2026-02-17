# import requests
# from datetime import datetime

# def ask_llm(prompt):
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         # json={
#         #     "model": "llama3.1:8b",
#         #     "prompt": prompt,
#         #     "stream": False
#         # }

#         json={
#             "model": "phi3:mini",
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "num_predict": 60,
#                 "temperature": 0.7
#             }
#         }
#     )
#     return response.json()["response"]

# while True:
#     #concat readable timestamps to right in every
#     user = input("You: ")
#     # print user's message with readable timestamp to the right
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"You: {user}    {ts}")

#     reply = ask_llm(user)
#     # print agent reply with readable timestamp to the right
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"Agent: {reply}    {ts}")

import requests
import json

def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": True,
            "options": {
                # "num_predict": 50,
                "temperature": 0.7
            }
        },
        stream=True
    )

    full_response = ""

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            print(token, end="", flush=True)
            full_response += token

    print()
    return full_response


while True:
    user = input("You: ")
    print("Agent: ", end="")
    ask_llm(user)
