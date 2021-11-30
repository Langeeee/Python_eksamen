from os import listdir
from os.path import isfile, join
class GetFiles:

    def __int__(self):
        print()

    def getFilesAsList(self, filePath):
        onlyfiles = [f for f in listdir(filePath) if isfile(join(filePath, f))]
        return onlyfiles

    def getFileValues(self, fileList):
        valueList = []
        for f in fileList:
            nameAndValue = f.split(".")
            valueList.append(nameAndValue[0])
        return valueList