import requests

class DownloadAndSave:

    def __init__(self):
        print("OK")

    def makeDownload(self, URL, filename):
        data = requests.get(URL, allow_redirects=True)
        local_file = filename

        with open(local_file, 'wb')  as file:
            file.write(data.content)
    