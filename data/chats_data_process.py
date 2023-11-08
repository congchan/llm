# -*- coding: utf-8 -*-

import collections
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
from constants import USER, ASSIS, RP_SYSTEM, MTRP_SYSTEM

from utils import read_text


"""['AFRIKAANS', 'ALBANIAN', 'ARABIC', 'ARMENIAN', 'AZERBAIJANI', 'BASQUE', 'BELARUSIAN', 'BENGALI', 'BOKMAL', 
'BOSNIAN', 'BULGARIAN', 'CATALAN', 'CHINESE', 'CROATIAN', 'CZECH', 'DANISH', 'DUTCH', 'ENGLISH', 'ESPERANTO', 
'ESTONIAN', 'FINNISH', 'FRENCH', 'GANDA', 'GEORGIAN', 'GERMAN', 'GREEK', 'GUJARATI', 'HEBREW', 'HINDI', 'HUNGARIAN', 
'ICELANDIC', 'INDONESIAN', 'IRISH', 'ITALIAN', 'JAPANESE', 'KAZAKH', 'KOREAN', 'LATIN', 'LATVIAN', 'LITHUANIAN', 
'MACEDONIAN', 'MALAY', 'MAORI', 'MARATHI', 'MONGOLIAN', 'NYNORSK', 'PERSIAN', 'POLISH', 'PORTUGUESE', 'PUNJABI', 
'ROMANIAN', 'RUSSIAN', 'SERBIAN', 'SHONA', 'SLOVAK', 'SLOVENE', 'SOMALI', 'SOTHO', 'SPANISH', 'SWAHILI', 'SWEDISH', 
'TAGALOG', 'TAMIL', 'TELUGU', 'THAI', 'TSONGA', 'TSWANA', 'TURKISH', 'UKRAINIAN', 'URDU', 'VIETNAMESE', 'WELSH', 
'XHOSA', 'YORUBA', 'ZULU'] """
LANGUAGES = [
    Language.ENGLISH,
    Language.CHINESE,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.JAPANESE,
    Language.ITALIAN,
]

LANG_DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

LANG_MAPPING = {
    "Language.ENGLISH": "en",
    'Language.CHINESE': "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "None": "other_langs",
}

ID2NAME = (
    ("en", "英语"),
    ("fr", "法语"),
    ("es", "西班牙语"),
    ("Language.ENGLISH", "英语"),
    ("Language.ENGLISH", "英国"),
    ("Language.ENGLISH", "美国"),
    ("Language.ENGLISH", "英国"),
    ("Language.FRENCH", "法语"),
    ("Language.FRENCH", "法国"),
    ("Language.SPANISH", "西班牙语"),
)


def get_turns(item):
    turns = item.get("conversations") if "conversations" in item else item.get("conversation", [])
    return turns


def get_role_and_content(turn, role_key="from", content_key="value"):
    from_key = None
    value_key = None

    for _key in ["from", "role", "speaker"]:
        if _key in turn:
            from_key = _key

    for _key in ["value", "text", "content", "sentence", "words"]:
        if _key in turn:
            value_key = _key
    
    if from_key is None or value_key is None:
        return None
    
    value = turn[value_key].strip()
    if not value:
        return None
        
    _formatted_turn = {content_key: value}
    
    if turn[from_key] in (
        ASSIS, "Assistant", "ASSISTANT", "assistant", "GPT", "gpt", "gpt4", "chatgpt", "AI", "机器人", "助手",  "MEI",
    ):
        _formatted_turn[role_key] = ASSIS
        _formatted_turn["type"] = "bot_query"
        
    elif turn[from_key] in (
        USER, "Human", "HUMAN", "human", "User", "user", "USER", "用户", "人类"
    ):
        _formatted_turn[role_key] = USER
        _formatted_turn["type"] = "user_query"
    else:
        _formatted_turn[role_key] = turn[from_key]
        _formatted_turn["type"] = "user_query"
    
    return _formatted_turn


def get_session_list(raw_data, id_prefix, src, data_type="conversation"):
    """
    Process raw data to session data for finetuing.

    Args:
        raw_data (list): raw data list
        id_prefix (str): data id prefix
        src (str): data source
        data_type (str, optional): data type

    Returns:
        list: formatted session list
    """
    
    formated_samples = []
    for idx, item in tqdm(enumerate(raw_data), total=len(raw_data)):
        conv_ls = []
        
        turns = get_turns(item)
    
        if len(turns) < 2:
            print(f"{idx}: not a complete conversations, maybe missing keys |conversations|")
            print(item)
            continue

        is_valid = True
        for turn in turns:
            formatted_turn = get_role_and_content(turn)
            if formatted_turn is None:
                continue

            value = formatted_turn["value"]
            if formatted_turn["from"] == USER:
                if len(conv_ls) > 0 and conv_ls[-1]['from'] == USER:
                    print(f"{idx} -> {len(formated_samples)}: Consecutive user turn: {value}\n")
                    is_valid = False

            elif formatted_turn["from"] == ASSIS:
                if len(conv_ls) > 0 and conv_ls[-1]['from'] == ASSIS:
                    print(f"{idx} -> {len(formated_samples)}: Consecutive assis turn: {value}\n")
                    is_valid = False

            else:
                print(f"{idx} Not sure who said this turn: {value}")
                is_valid = False

            conv_ls.append(formatted_turn)

        if len(conv_ls) > 0 and conv_ls[-1]["from"] == USER:
            print(f"{idx} -> {len(formated_samples)}: Pop out last user turn without respond.\n")
            conv_ls.pop()

        if len(conv_ls) < 2:
            print(f"{idx}: not a complete conversations")
            print(item)
            continue

        assert conv_ls[0]["from"] == USER

        if is_valid:
            new_sample = {
                'id': f'{id_prefix}_{str(idx)}',
                'conversations': conv_ls,
                'src': src,
                'type': data_type,
            }
            formated_samples.append(new_sample)

    return formated_samples


def worker_func(item, idx):
    """
    Parralel detecting language

    Args:
        item (dict): session data with 'conversations' as key
        idx (int): index

    Returns:
        tuple of detected lang, original data and index
    """
    # sanity check 
    if len(item['conversations']) < 2:
        return (None, None, idx)

    value_in = "".join([x['value'] for x in item['conversations'] if x['from'] == 'human'])
    value_out = "".join([x['value'] for x in item['conversations'] if x['from'] == 'gpt'])

    lang_in = str(LANG_DETECTOR.detect_language_of(value_in))
    lang_out = str(LANG_DETECTOR.detect_language_of(value_out))

    if lang_in in ('Language.CHINESE', 'zh') and lang_out != lang_in:
        conditions = [
            any(x in value_in for x in ["翻译成中文", "翻译为中文"]),
            any(lang_out == lang_id and lang_name not in value_in for (lang_id, lang_name) in ID2NAME),
            (not value_out),
        ]

        if any(conditions):
            return (None, None, idx)

    return (
        lang_out if lang_out in (
        'en', 'Language.ENGLISH', 'zh', "zh-cn", "zh-tw", 'Language.CHINESE') else 'other_langs',
        item,
        idx
    )


def concurrent_detect_lang_as_completed(data, max_workers):
    """
    Concurrently detects the language in the data and stores the detection results in the data.

    Args:
        ata (list): A list of data to be detected, 
                    each element is a dictionary containing the text to be detected and other information
        max_workers (int): Maximum number of processes

    Returns:
        None
    """
    import concurrent.futures
    import time
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = [executor.submit(worker_func, item, idx) for idx, item in enumerate(data)]
        print("Running jobs", flush=True)
        for result in tqdm(concurrent.futures.as_completed(results), total=len(results)):
            lang, item, idx = result.result()
            if lang and item:
                data[idx]["detected_lang"] = lang if lang in (
                'en', 'Language.ENGLISH', 'zh', "zh-cn", "zh-tw", 'Language.CHINESE') else 'other_langs'

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("concurrent cost time: ", elapsed_time, "seconds", f"Speed: {len(data) / elapsed_time} samples/s")
