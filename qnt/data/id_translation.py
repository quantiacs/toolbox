import os
import csv
import typing as tp

USE_ID_TRANSLATION: bool = True
TRANSLATION_TABLE_FILE_NAME = "id-translation.csv"

server_id_to_user_id = None
user_id_to_server_id = None


def translate_asset_to_user_id(asset):
    if not USE_ID_TRANSLATION:
        return asset['id']
    return get_or_create_translation(asset['id'], asset.get('exchange', 'GLOBAL') + ':' + asset['symbol'])


def load_id_translation_table():
    global server_id_to_user_id, user_id_to_server_id
    if server_id_to_user_id is not None:
        return
    server_id_to_user_id = dict()
    user_id_to_server_id = dict()
    if os.path.exists(TRANSLATION_TABLE_FILE_NAME):
        with open(TRANSLATION_TABLE_FILE_NAME) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            headers = None
            for row in csv_reader:
                for i in range(len(row)):
                    row[i] = row[i].strip()
                if headers is None:
                    headers = row
                else:
                    if len(row) == 2:
                        entry = dict(zip(headers, row))
                        server_id_to_user_id[entry['server_id']] = entry['user_id']
                        user_id_to_server_id[entry['user_id']] = entry['server_id']


def get_or_create_translation(server_id, preferred_user_id):
    if not USE_ID_TRANSLATION:
        return server_id

    load_id_translation_table()

    if server_id_to_user_id.get(server_id) is not None:
        return server_id_to_user_id[server_id]

    if 'SUBMISSION_ID' in os.environ and os.environ['SUBMISSION_ID'] != '':
        return server_id

    num = 0
    user_id = preferred_user_id
    while user_id_to_server_id.get(user_id) is not None:
        num += 1
        user_id = preferred_user_id + "~" + str(num)

    if not os.path.exists(TRANSLATION_TABLE_FILE_NAME):
        with open(TRANSLATION_TABLE_FILE_NAME, 'w') as file:
            file.write("server_id,user_id\n")

    with open(TRANSLATION_TABLE_FILE_NAME, 'a') as file:
        file.write(server_id + "," + user_id + "\n")

    server_id_to_user_id[server_id] = user_id
    user_id_to_server_id[user_id] = server_id

    return user_id


def translate_user_id_to_server_id(user_id):
    if not USE_ID_TRANSLATION:
        return user_id
    load_id_translation_table()
    return user_id_to_server_id.get(user_id, user_id)


def translate_server_id_to_user_id(server_id):
    if not USE_ID_TRANSLATION:
        return server_id
    load_id_translation_table()
    return server_id_to_user_id.get(server_id, server_id)
