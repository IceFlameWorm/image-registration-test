from schema import Schema, And, Use, Optional,Or

ocr_external_result_schema = Schema({
    "page":{
        "pageNo":Use(int),
        "matched": bool,
        "punchholeBoundingBoxList": [[int]],
        "studentInfo": {
            "studentId": str,
            "studentIdBoundingBox": [int],
            "studentIdCandidateInfo": [
                {
                    "best_dist": int,
                    "best_score": float,
                    "best_valid_id": str,
                    "student_number_str": str,
                    "matching_info": {
                        Optional(str): {
                            "dist": int,
                            "score": float,
                            "student_number_str": str,
                            "valid_id": str
                        }
                    }
                }
            ],
            "studentName": str,
            "studentNameBoundingBox": [int]
        },
        "completionTime": int,
        "questionList": [
            {
                "meta": {
                    "matched": bool,
                    "similarity": float,
                    "questionId": int,
                },
                "boundingBoxList": [[int]],
                "questionSubList": [
                    {
                        "meta": {
                            "matched": bool,
                            "similarity": float,
                            "bookPageQuestionId": int,
                            "question_type": str
                        },
                        "boundingBoxList": [[int]],
                        "blankList": [
                            {
                                "boundingBoxList": [[int]],
                                "segmentList": [
                                    {
                                        "contentList": [str],
                                        "confidenceList": [float],
                                        "boundingBox": [int],
                                        "idx": int
                                    }
                                ],
                                "selectedOptionList": [str],
                                "extendInfo": {
                                    "trustworthy": {
                                        'type': int,
                                        'code': int,
                                        'confidence': float,
                                        'msg': str
                                    },
                                    "uid": str,
                                    "union_answer_uids": [str],
                                    "candidate_segments": [
                                        {
                                            "contentList": [str],
                                            "confidenceList": [float],
                                            "boundingBox": [int],
                                            "idx": int
                                        }
                                    ]
                                }
                            }
                        ],
                        "typewrittenList": [
                            {
                                "contentList": [str],
                                "confidenceList": [float],
                                "boundingBox": [int],
                                "idx": int
                            }
                        ],
                        "sketchList": [
                            {
                                "contentList": [str],
                                "confidenceList": [float],
                                "boundingBox": [int],
                                "idx": int
                            }
                        ],
                        "studentDiagramList": [],
                        "extendInfo": {
                            "trustworthy": {
                                'type': int,
                                'code': int,
                                'confidence': float,
                                'msg': str
                            }
                        }
                    }
                ]
            }
        ]                         
        }
    }
, ignore_extra_keys=True)