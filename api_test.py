import time

from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)
t1 = time.time()
completion = client.chat.completions.create(
  extra_body={},
  model="mistralai/mistral-small-3.1-24b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Что на картинке?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"
          }
        }
      ]
    }
  ]
)
print(time.time()-t1)
print(completion.choices[0].message.content)