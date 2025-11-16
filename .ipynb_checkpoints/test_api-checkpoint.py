
import requests

# آدرس API که Flask در ترمینال نشون داده بود
url = "http://127.0.0.1:5000/predict"

# داده‌ای که می‌خوای برای تست بفرستی (ورودی مدل)
data = {
    "area": 100,
    "rooms": 3,
    "distance": 10
}

# ارسال درخواست POST به سرور
response = requests.post(url, json=data)

# نمایش پاسخ مدل (قیمت پیش‌بینی شده)
print(response.json())
