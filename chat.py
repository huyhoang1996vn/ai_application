# https://platform.openai.com/docs/api-reference/chat/create
from openai import OpenAI
from main import API_KEY
client = OpenAI(api_key=API_KEY)

# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[
#     {"role": "developer", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)

# from openai import OpenAI
# client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! where are you from"}
  ],
  stream=True
)

for chunk in completion:
  print(chunk.choices[0].delta.content or "", end="", flush=True)
