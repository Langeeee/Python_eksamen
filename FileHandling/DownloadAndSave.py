import requests
import base64
import re

class DownloadAndSave:

    def __init__(self):
        print("OK")
    
    

    def makeDownload(self, URL, filename):
        def decode_base64(data, altchars=b'+/'):
            data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
            missing_padding = len(data) % 4
            if missing_padding:
                data += b'='* (4 - missing_padding)
            return base64.b64decode(data, altchars)
        #data = requests.get(URL, allow_redirects=True)
        fileString = URL[22:-1]
        print(fileString)
        local_file = filename
        arr = bytes(fileString, encoding='utf8')
        with open(local_file, 'wb')  as file:
            file.write(decode_base64(arr))
    
    