import os
import sys
import re
current_dir = os.getcwd()
print("Current DIR : {}\n".format(current_dir))
sys.path.insert(0, os.path.abspath(f"{current_dir}/lib/python3.10/site-packages"))
print("path : {}\n".format(sys.path))

from datetime import datetime
from complaince.mobito2_0.utils import (run_config_reader,
                                         create_status_json, basic_validator_and_parser,
                                         send_status_to_mobito, get_processed_ip)
from complaince.mobito2_0.aws_utils import S3_downloader, S3_Upload
from functionality.news_segmentation import segment

def main():

    pid = str(os.getpid())
    bot_start_time = re.sub('\.[^<]*?$', '', str(datetime.now()))

    input_json_filename, input_json, input_id, OutputId, mob_id, clientid, ref_id, in_id, user_attr \
        = basic_validator_and_parser(sys.argv)
    
    config_json = run_config_reader()
    MOBITO_ENDPOINT = config_json['BOT']['BOT_URL']

    #local_input_path = S3_downloader(user_attr['s3_input_stem'], user_attr['input_file_name'])

    send_status_to_mobito('INITIATED', ref_id, pid, input_json_filename, MOBITO_ENDPOINT)

    #Core logic
    JSON_MSG = {"s3_input_stem" : user_attr['s3_input_stem'], "input_file_name" : user_attr['input_file_name']}
    data = segment(user_attr['endpoint'], JSON_MSG)

    bot_end_time = re.sub('\.[^<]*?$', '', str(datetime.now()))
    processed_ip = get_processed_ip()

    result = {
        'bot_start_time' : bot_start_time,
        'bot_end_time' : bot_end_time,
        'bot_name' : 'Newspaper Segmentation v1.0',
        'version' : '1.0',
        'processed_ip' : processed_ip,
        **input_json,
        'op' : [],
        'op_json_status' : 'False',
        'process_final_status' : 'Success',
        'status_scenario' : ''
    }
    if not isinstance(data, type([])):
        error = {'error_message' : 'Unknown Exception', 'error_code' : '1004'}
        op_path = create_status_json(ref_id, processed_ip, bot_start_time, bot_end_time, error, 
                                     input_json, {**result, **error})
        dest = f"{user_attr['s3_input_stem']}".replace('outputs', 'recognized_outputs')+(op_path.parts[-1])
        upload_result = S3_Upload(op_path, dest)
        send_status_to_mobito('ERROR', ref_id, pid, input_json_filename, MOBITO_ENDPOINT)
    else:
        error = {'error_message' : '', 'error_code' : ''}
        result['op'] = data
        result['op_json_status'] = 'True'
        op_path = create_status_json(ref_id, processed_ip, bot_start_time, bot_end_time, error, 
                                        input_json, {**result, **error})
        dest = f"{user_attr['s3_input_stem']}".replace('outputs', 'recognized_outputs')+(op_path.parts[-1])
        upload_result = S3_Upload(op_path, dest)
        if isinstance(upload_result, type(True)):
            send_status_to_mobito('COMPLETED', ref_id, pid, input_json_filename, MOBITO_ENDPOINT, len(data))
        else:
            result['op'] = data
            error = {'error_message' : 'Unable to upload to S3 bucket', 'error_code' : '1004'}
            op_path = create_status_json(ref_id, processed_ip, bot_start_time, bot_end_time, error, 
                                        input_json, {**result, **error})
            send_status_to_mobito('ERROR', ref_id, pid, input_json_filename, MOBITO_ENDPOINT, len(data))
if __name__ == '__main__':
    main()
    




