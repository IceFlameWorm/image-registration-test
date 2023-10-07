from schema import Schema, And, Use, Optional,Or

# todo: index/order should be not None!
position_info_schema = Schema([
    {
        'pageNo': And(int, lambda x: x >= 0, error='pageNo must be positive interger.'),
        'pageUrl': And(str, lambda x: len(x.strip()) > 0 and x.startswith('http'), error='pageUrl must be validate http url.'),
        'positionType': And(str, lambda x: len(x.strip()) > 0),
        'templateImgInfo': {
            'rect': {
                'startX': And(int, lambda x: x >= 0),
                'startY': And(int, lambda x: x >= 0),
                'endX': And(int, lambda x: x >= 0),
                'endY': And(int, lambda x: x >= 0),
                'width': And(int, lambda x: x >= 0),
                'height': And(int, lambda x: x >= 0)
            }
        },
        'anchorList': [
            {
                'rect': {
                    'startX': And(int, lambda x: x >= 0),
                    'startY': And(int, lambda x: x >= 0),
                    'endX': And(int, lambda x: x >= 0),
                    'endY': And(int, lambda x: x >= 0),
                    'width': And(int, lambda x: x >= 0),
                    'height': And(int, lambda x: x >= 0)
                }
            }
        ],
        'qrcodeList': [
            {
                'rect': {
                    'startX': And(int, lambda x: x >= 0),
                    'startY': And(int, lambda x: x >= 0),
                    'endX': And(int, lambda x: x >= 0),
                    'endY': And(int, lambda x: x >= 0),
                    'width': And(int, lambda x: x >= 0),
                    'height': And(int, lambda x: x >= 0)
                }
            }
        ],
        'columnList': [
            {
                'rect': {
                    'startX': And(int, lambda x: x >= 0),
                    'startY': And(int, lambda x: x >= 0),
                    'endX': And(int, lambda x: x >= 0),
                    'endY': And(int, lambda x: x >= 0),
                    'width': And(int, lambda x: x >= 0),
                    'height': And(int, lambda x: x >= 0)
                }
            }
        ],
        'questionList': [
            {
                'questionId': And(int, lambda x: x >= 0, error='questionId must be positive interger.'),
                'blocks': Or(None, [
                    {
                        'type': str,
                        'rect': {
                            'startX': And(int, lambda x: x >= 0),
                            'startY': And(int, lambda x: x >= 0),
                            'endX': And(int, lambda x: x >= 0),
                            'endY': And(int, lambda x: x >= 0),
                            'width': And(int, lambda x: x >= 0),
                            'height': And(int, lambda x: x >= 0),
                            'index': Or(And(int, lambda x: x>=1), None)
                        }
                    }
                ]),
                'subQuestionList': [
                    {
                        'bookPageQuestionId': And(int, lambda x: x >= 0, error='bookPageQuestionId must be positive interger.'),
                        'blocks': [
                            {
                                'type': str,
                                'rect': {
                                    'startX': And(int, lambda x: x >= 0),
                                    'startY': And(int, lambda x: x >= 0),
                                    'endX': And(int, lambda x: x >= 0),
                                    'endY': And(int, lambda x: x >= 0),
                                    'width': And(int, lambda x: x >= 0),
                                    'height': And(int, lambda x: x >= 0),
                                    'index': Or(And(int, lambda x: x>=1), None)
                                }
                            }
                        ],
                        'blankList': Or(None, [
                            {
                                'bookPageQuestionId': And(int, lambda x: x >= 0, error='bookPageQuestionId must be positive interger.'),
                                'order': Or(None, And(int, lambda x: x>= 1)),
                                'type': And(str, lambda x: len(x.strip()) > 0),
                                'blocks': [
                                    {
                                        'type': Or(str, None),
                                        'rect': {
                                            'startX': And(int, lambda x: x >= 0),
                                            'startY': And(int, lambda x: x >= 0),
                                            'endX': And(int, lambda x: x >= 0),
                                            'endY': And(int, lambda x: x >= 0),
                                            'width': And(int, lambda x: x >= 0),
                                            'height': And(int, lambda x: x >= 0),
                                            'index': Or(And(int, lambda x: x>=1), None)
                                        }
                                    }
                                ]
                            }
                        ])
                    }
                ]
            }
        ]
    }
], ignore_extra_keys=True)

# raise Exception('data format/info is invalidate!')
def input_logic_check(question_info_list, page_pos_info_list):
    subquestion_map = {}
    for question_info in question_info_list:
        question_id = question_info['questionId']
        for subquestion_info in question_info['questionSubList']:
            subquestion_id = subquestion_info['bookPageQuestionId']
            page_no = subquestion_info['pageNo']
            blank_count = subquestion_info['blankCount']
            subquestion_map[subquestion_id] = {
                'question_id': question_id,
                'subquestion_id': subquestion_id,
                'page_no': page_no,
                'blank_count': blank_count,
            }
    
    subquestion_pos_map = {}
    for page_pos_info in page_pos_info_list:
        page_no = page_pos_info['pageNo']
        for question_pos_info in page_pos_info['questionList']:
            question_id = question_pos_info['questionId']
            for subquestion_pos_info in question_pos_info['subQuestionList']:
                subquestion_id = subquestion_pos_info['bookPageQuestionId']
                blank_count = len(subquestion_pos_info['blankList'] or [])
                if subquestion_id in subquestion_pos_map:
                    subquestion_pos_map[subquestion_id]['blank_count'] += blank_count
                else:
                    subquestion_pos_map[subquestion_id] = {
                        'question_id': question_id,
                        'subquestion_id': subquestion_id,
                        'page_no': page_no,
                        'blank_count': blank_count,
                    }

    if len(subquestion_map) != len(subquestion_pos_map):
        raise Exception('len(subquestion_map) != len(subquestion_pos_map)')
    for subquestion_id, info in subquestion_map.items():
        pos = subquestion_pos_map.get(subquestion_id, None)
        if pos is None:
            raise Exception(f'position for subquestion is None, subquestion_id: {subquestion_id}')
        if info != pos:
            raise Exception(f'question_id/page_no/blank_count for info and pos are different! info: {info}, pos: {pos}')


