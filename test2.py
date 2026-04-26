import base64

def obscure_key(key):
    # 1. Переворачиваем строку
    reversed_key = key[::-1]
    # 2. Кодируем в Base64
    encoded = base64.b64encode(reversed_key.encode()).decode()
    return encoded

original_key = "AIzaSyBmznJqRZzWZTlIQJ8Gb9ye7ppHrTxcZ5w"
print(f"Зашифрованный ключ: {obscure_key(original_key)}")