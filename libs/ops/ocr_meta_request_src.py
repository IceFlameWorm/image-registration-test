import requests

# headers = {"accessToken": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhaSIsImF1ZCI6IjEzMjE2NDg3Njc2MzEzNyIsImV4cCI6MTYxNzI1OTAzOH0.pxvO26OZhsXu_i2Ilu5_5blhrqj5u8xmNQ7INdivBzs"}
# base_url = "http://api.qb.kezhitech.com/question-domain"

def get_all_pages_of_given_eb(eb_id,base_url,headers):
    """
    this function gets all the page numbers in this exercise book
    """
    url_que  = '{}/exerciseBook/queryAllPageNo?exerciseBookId={}'.format(base_url,eb_id)
    page_nos = requests.get(url_que, headers=headers).json()
    return page_nos

def get_question_info(que_id,base_url,headers):
    """
    note: this api originally returns a lot of information related to questions
    for matching purposes, many keys are filtered out.
    Please see the whitelist variable below to see the specific meanings of each key
    """

    url = "{}/question/find?questionId={}".format(base_url,que_id)
    question_info = requests.get(url, headers=headers).json()["data"]
    
    """
    ======== sanitize the question keys =======
    """
    que_whitelist = [
        'questionBody',  # this is the parent question body. We match the parent question using this key
        'questionSubject', # subject coding api coming soon
        'questionType', # question type coding api coming soon
        "questionSubList", # this points to the subquestions
        'questionId',
        'bookPageQuestionId',
    ]
    question_info_keys = list(question_info.keys())
    for k in question_info_keys:
        if k not in que_whitelist:
            question_info.pop(k)

    """
    ======== sanitize the sub question keys =======
    """
    sub_que_whitelist = [
        'reviewQuestionType',
        'conditionInfo', # this, concatenated with problem info, can be seen as the sub question body. We match the sub question using this key
        'problemInfo',
        'otherInfo',
        'questionSubId',
        'questionSubIdx',
        'blankCount'

    ]
    for sub_que in question_info["questionSubList"]:
        sub_question_info_keys = list(sub_que.keys())
        for k in sub_question_info_keys:
            if k not in sub_que_whitelist:
                sub_que.pop(k)

    return question_info

def get_all_question_ids_specific_page_of_given_eb(eb_id, page_no,base_url,headers):
    """
    this function gets all the question ids on this page of the given exercise book
    """
    url = "{}/exerciseBook/queryAllQuestionListByBookIdAndPageNo?exerciseBookId={}&pageNo={}".format(base_url,eb_id,page_no)
    question_ids = [i["questionId"] for i in requests.get(url, headers=headers).json()["data"]]
    return question_ids

def get_all_question_info_specific_page_of_given_eb(eb_id, page_no,base_url,headers):
    question_ids = get_all_question_ids_specific_page_of_given_eb(eb_id, page_no,base_url,headers)
    return [get_question_info(i,base_url,headers) for i in question_ids]


