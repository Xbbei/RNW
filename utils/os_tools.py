import os


def get_files(d, postfix=None):
    """
    Return file names in a given directory with given postfix
    :param d:
    :param postfix:
    :return:
    """
    files = []
    for f in os.scandir(d):
        if f.is_file():
            if postfix is None or f.name.endswith(postfix):
                files.append(f.name)
    return files
