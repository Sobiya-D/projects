from .logging import log
from configs.path import PARENT_DIR, CONFIG_DIR
from pathlib import Path
import requests
import yaml
import json
import socket
import public_ip

@log()
def run_config_reader(filename : str = 'config.yaml'):
    input_json_path = CONFIG_DIR /  filename
    with open(input_json_path, 'r') as f:
        config_json  =  yaml.safe_load(f)
    return config_json

@log()
def create_status_json(ref_id : str, processed_ip : str, start : str, end : str,
                        error : dict, input_ : dict, result : dict):
    output_template = {
    "ref_id" : ref_id,
    "Processed_IP" : processed_ip,
    "bot_start_time" : start,
    "bot_end_time" : end,
    "error" : error,
    "Input" : input_,
    "Result" : {"Output" : [result]}
    }
    output_name = PARENT_DIR / f'Output_{str(len(result["op"]))}_{ref_id}.json'
    with open(output_name, 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(output_template, indent = 4, ensure_ascii=False))
    return output_name

@log()
def basic_validator_and_parser(args):
    if len(args) != 2:
        raise Exception('Missing/More arguments')
    input_file = Path(args[1])
    if not input_file.is_file():
        raise Exception('Input JSON file unavailable')
    with open(input_file, 'r') as f:
        input_json = json.load(f)
    if not 'bot_input' in input_json:
        raise Exception('bot_input JSON key missing/differs')
    input_id = input_json['bot_input'][0]['InputId']
    OutputId = input_json['bot_input'][0]['OutputId']
    mob_id = input_json['bot_input'][0]['mobid']
    clientid = input_json['bot_input'][0]['clientid']
    ref_id = input_json['bot_input'][0]['ref_id']
    in_id = input_json['bot_input'][0]['in_id']
    user_attr = set(input_json['bot_input'][0].keys()) - \
        {'InputId', 'OutputId', 'mobid', 'clientid', 'ref_id', 'in_id'}
    user_attr = {key: input_json['bot_input'][0][key] for key in user_attr}
    return (args[1], input_json['bot_input'][0], input_id,\
             OutputId, mob_id, clientid, ref_id, in_id, user_attr)

@log()
def send_status_to_mobito(typ : str, ref_id : str, pid : str,
                         input_json_filename : str, mobito_endpoint : str, result : int = 0):
    if typ in ('ERROR', 'COMPLETED'):
        input_json_filename = ''
        output_json_filename = f'Output_{result}_{ref_id}.json'
    elif typ == 'INITIATED':
        ref_id = ref_id
        output_json_filename = ''
    mobito_status = {}
    mobito_status["ref_id"] = ref_id
    mobito_status["pid"] = pid
    mobito_status["input_json_filename"] = input_json_filename
    mobito_status["status"] = typ
    mobito_status["output_json_filename"] = output_json_filename
    mobito_status = json.dumps(mobito_status, indent=4)
    ping_status_to_mobito({'bot-status' : mobito_status}, mobito_endpoint)

@log(raise_error = False)
def _post_to_mobito(mobito_status, mobito_endpoint):
    global MOBITO_ENDPOINT_CONNECTED
    requests.post(mobito_endpoint, data = mobito_status)
    MOBITO_ENDPOINT_CONNECTED = True

@log()
def ping_status_to_mobito(mobito_status : dict, mobito_endpoint : str, n_times : int = 4):
    global MOBITO_ENDPOINT_CONNECTED
    MOBITO_ENDPOINT_CONNECTED = False
    tries = 0
    while not MOBITO_ENDPOINT_CONNECTED and tries + 1 <= n_times:
        _post_to_mobito(mobito_status, mobito_endpoint)
        tries += 1

@log()
def get_processed_ip():
    if (socket.gethostbyname(socket.gethostname())):
        return socket.gethostbyname(socket.gethostname())
    else:
        return public_ip.get()