import random
import time
from g4f.client import ClientFactory

client = ClientFactory.create_client("custom:srv_mkoloq41e34074b6133e", api_key="g4f_u_mnp0kf_a3b5287dd5d041ba21d53947a33342e9b92f496b808e9b43_02debd59")
while True:
    t1 = time.time()
    stream  = client.chat.completions.create(
        model="gemini-fast",
        messages=[{"role": "user", "content": f"Привет"}],
        stream=True
    )
    is_first = True
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if is_first:
                print(time.time() - t1)
                is_first = False
            print(chunk.choices[0].delta.content or "", end="")
    print()