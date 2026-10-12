"""Self-check для qr_wifi: парсер WIFI:-QR не должен зависеть от порядка полей.

Run:  .venv/bin/python test_qr_wifi.py
"""

from robov_core.qr_wifi import parse_wifi_qr


def main() -> None:
    # канонический порядок (T,S,P) — работал и старым регексом
    assert parse_wifi_qr("WIFI:T:WPA;P:secret;S:Home;;") == {
        "auth": "WPA", "ssid": "Home", "password": "secret"}

    # порядок телефона (S первым) + поле H + лишние ';'
    r = parse_wifi_qr("WIFI:S:Home;T:WPA;P:secret;H:true;;;")
    assert r == {"auth": "WPA", "ssid": "Home", "password": "secret"}, r

    # открытая сеть без P
    r = parse_wifi_qr("WIFI:T:nopass;S:Cafe;;")
    assert r == {"auth": "nopass", "ssid": "Cafe", "password": ""}, r

    # экранированные \; и \: внутри ssid/пароля
    r = parse_wifi_qr(r"WIFI:T:WPA;S:Co\;ffee;P:pa\:ss;;")
    assert r == {"auth": "WPA", "ssid": "Co;ffee", "password": "pa:ss"}, r

    # не WIFI:-QR / нет ssid
    assert parse_wifi_qr("https://example.com") is None
    assert parse_wifi_qr("WIFI:P:only;;") is None

    print("qr_wifi self-check OK")


if __name__ == "__main__":
    main()
