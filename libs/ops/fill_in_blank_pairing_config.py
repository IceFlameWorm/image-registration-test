import re
import copy

class QUENO_Config:
    que_nos = [str(x) for x in range(100)]
    que_no = re.compile(r"(\d{1,2})")
    hacky_convert = {}
    replace_dict = {" ":"",".": "", ":": "", "\\]": "1", "\\[": "1", "\\mid": "1", "i": "1",  "-": "", "!": "1"}
    invalid_chars = "分","共","题","小","大"

class LS_01_process(QUENO_Config):
    replace_dict = copy.deepcopy(QUENO_Config.replace_dict)
    # replace_dict.update({"7": "1"})


class GK_05_process(QUENO_Config):
    que_nos = [str(i) for i in range(1, 15)]
    que_no = re.compile(r"(\d{1,2})")


class ZK2_process(QUENO_Config):
    que_nos = ["11", "12", "13", "14", "15", "16"]


class ZK1_process(QUENO_Config):
    que_nos = ["13", "14", "15", "16", "17", "18"]


class ZK3_process(QUENO_Config):
    que_nos = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    que_no = re.compile(r"(\d)")



def extract_que_no(contents, config: QUENO_Config):
    """
    return: str, if the string is "-1", it means no question number was matched
    """
    que_no = "-1"
    
    for kk in contents:
        for invalid_char in config.invalid_chars:
            if invalid_char in kk:
                return "-1"
        for o, p in config.replace_dict.items():
            kk = kk.replace(o, p)
        found_que_no = config.que_no.findall(kk)
        if found_que_no and str(found_que_no[0]) in config.que_nos:
            que_no = int(found_que_no[0])
            que_no = config.hacky_convert[que_no] if que_no in config.hacky_convert else que_no
            break
    return str(que_no)

