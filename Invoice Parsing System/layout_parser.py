from complaince.mobito2_0.logging import log
import requests

@log()
def parse_invoice(endpoint, JSON_MSG, headers):
    try:
        response = requests.post(endpoint, json = JSON_MSG, headers=headers, timeout = 12)
        if response.status_code == 200:
            return 200, response.json()['data']
    except:
        return 404, 'Unable to hit the server!'

@log()
def recognize(endpoint, JSON_MSG):
    try: 
        headers = {"charset": "utf-8", "Content-Type": "application/json"}
        attempts = 1
        while attempts:
            code, msg = parse_invoice(endpoint, JSON_MSG, headers)
            if code == 200:
                return msg
            else:
                attempts -= 1
        return 'Unable to hit the server!'
    except Exception as e:
        return None