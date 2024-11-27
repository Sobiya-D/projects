from complaince.mobito2_0.logging import log
import requests

@log()

def segment(endpoint, JSON_MSG):
    try: 
        headers = {"charset": "utf-8", "Content-Type": "application/json"}
        response = requests.post(endpoint, json = JSON_MSG, headers=headers)
        if response.status_code == 200:
            return response.json()['data']
        else:
            return 'Failed'
    except Exception as e:
        print(e)
        return None