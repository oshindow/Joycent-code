import json

class ZHDict:
    def __init__(self, file_or_path):

        with open(file_or_path, 'r', encoding='utf8') as f:
            entries = json.load(f)
        
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        return int(self._entries.get(word))