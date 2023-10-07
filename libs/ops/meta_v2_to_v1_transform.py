""" ops for transform v2 meta to v1 meta for 
    rg enhanced question detection
    Copyright 2022, Learnable, inc , All rights reserved
"""




def extract_v2_rawquestionbody(meta: dict) -> dict:
    """
    extract raw quesiton body dict from v2 meta for
    rg enhanced question detection currently only
    supports meta of one book input format: whatever v2 meta is
    output format: dict(dict(list(str))):
        x[page_no][question_id] = list of rawQuestionBody Strings
    """
    cur_page = 0
    out_dict = {}
    for question in meta:
        # pprint(question)
        if 'rawQuestionBody' in question.keys():
            if question['rawQuestionBody']:
                if not question['questionId'] in out_dict.setdefault(cur_page, {}):
                    out_dict[cur_page][
                        question['questionId']] = question['rawQuestionBody']
        if question['questionSubList']:
            for subquestion in question['questionSubList']:
                cur_page = subquestion['pageNo']
                if 'rawQuestionBody' in subquestion.keys():
                    if subquestion['rawQuestionBody']:
                        cur_page = subquestion['pageNo']
                        out_dict.setdefault(cur_page, {})[
                            subquestion['bookPageQuestionId']] = \
                                subquestion['rawQuestionBody']
    return out_dict


def compose_fake_v1_meta_from_rawqestionbody_dict(raw_question_body_dict: dict) -> list:
    """
    compose fake v1 meta from composed raw question body dict
    input format: dict(dict(list(str))):
    x[page_no][question_id] = list of rawQuestionBody Strings
    output format: list(dict)
    """
    fake_raw_question = []
    global_id = 0
    for page,page_info in raw_question_body_dict.items():
        for question_id,raw_question_body in page_info.items():
            v1_question_dict = {'questionId': question_id,
                                'questionSubId':global_id+1,
                                'questionSubIdx':global_id+2,
                                'rawQuestionBody':raw_question_body,
                                'pageNo':page
                               }
            fake_raw_question.append(v1_question_dict)
            global_id += 3
    return fake_raw_question

# def compose_fake_v1_meta_from_rawqestionbody_dict(raw_question_body_dict: dict) -> dict:
#     """
#     compose fake v1 meta from composed raw question body dict
#     input format: dict(dict(list(str))):
#     x[page_no][question_id] = list of rawQuestionBody Strings
#     """
#     fake_raw_question = {}
#     global_id = 0
#     for page,page_info in raw_question_body_dict.items():
#         fake_raw_question[page] = []
#         for question_id,raw_question_body in page_info.items():
#             v1_question_dict = {'questionId': question_id,
#                                 'questionSubId':global_id+1,
#                                 'questionSubIdx':global_id+2,
#                                 'rawQuestionBody':raw_question_body
#                                }
#             fake_raw_question[page].append(v1_question_dict)
#             global_id += 3
#     return fake_raw_question

