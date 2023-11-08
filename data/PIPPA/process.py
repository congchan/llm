from tqdm import tqdm
import pandas as pd
import sys
from constants import USER, ASSIS, DEMO_SAMPLE, PROCESS_DIR, RP_SYSTEM

import os

print(PROCESS_DIR)
os.makedirs(PROCESS_DIR, exist_ok=True)

def get_session_list(raw_data, id_prefix, src, data_type="conversation"):
    import re
    formated_samples = []
    for idx, item in tqdm(enumerate(raw_data), total=len(raw_data)):
        bot_name = item["bot_name"]
        system = RP_SYSTEM
        background = item['bot_description'] if item['bot_description'] else item["bot_definitions"]
        if bot_name:
            background = f"Background of {bot_name}:\n" + background

        conv_ls = []
        is_valid = True
        for turn_id, turn in enumerate(item['conversation']):
            
            value = turn['message'].strip()

                
            if turn["is_human"]:
                if not value or len(value) < 1:
                    break
                conv_ls.append(
                    {'from': USER, 'value': value, "type": "user_query"},
                )
            else:
                if not value or len(value) < 1:
                    conv_ls.pop()
                    break 
                    
                conv_ls.append(
                    {'from': bot_name, 'value': value, "type": "bot_query"},
                )


        if len(conv_ls) > 0 and conv_ls[-1]["from"] == USER:
            conv_ls.pop()

        if len(conv_ls) < 2:
            continue

        if is_valid:
            new_sample = {
                'id': f'{id_prefix}_{item["bot_id"]}_{str(item["submission_timestamp"])}',
                "system": system,
            }
            if background:
                new_sample = new_sample | {
                    "background": background,
                }
            
            new_sample = new_sample | {
                'conversations': conv_ls,
                'src': src,
                'type': data_type,
            }
            formated_samples.append(new_sample)


    return formated_samples

id_prefix = "PIPPA"
src = "PIPPA"
formated_samples = get_session_list(raw_json, id_prefix, src)

turn_stats = [len(conversations['conversations'])/2 for conversations in formated_samples]
print(pd.DataFrame(turn_stats).describe().set_axis([id_prefix], axis=1))

save_path = os.path.join(PROCESS_DIR, f"{id_prefix}_en_chats.json")
print(save_path)
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(formated_samples, ensure_ascii=False))
