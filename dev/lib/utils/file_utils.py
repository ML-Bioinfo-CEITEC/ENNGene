# module containing methods for file handling


def filehandle_for(filename):
    if filename == "-":
        filehandle = sys.stdin
    else:
        filehandle = open(filename)
    return filehandle


