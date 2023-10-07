# {
# 2    "image_id": "4efaf068-bdae-4ad0-9f52-adb42270b06a",
# 3    "status": 0,
# 4    "timestamp": 122231,
# 5    "version": "sdfsd"
# 6    "result": {
# 7        "page": {
# 8            "pageNo": 12,
# 9            "punchhole_bbox": [1,2,3,4],
# 10            "student_info": {
# 11                "student_id": "123",
# 12                "student_id_bbox": [1,2,3,4],
# 13                "student_name_bbox": [1,2,3,4]
# 14            },
# 15            "completion_time": 13,
# 16            "questions": [
# 17                {
# 18                    "meta": {
# 19                        "matched": true,
# 20                        "similarity": 0.8,
# 21                        "questionId": 1125
# 22                    },
# 23                    "bbox": [[1,2,3,4], [1,2,3,4]],
# 24                    "questionSubList": [
# 25                        {
# 26                            "meta": {
# 27                                "matched": true,
# 28                                "similarity": 0.8,
# 29                                "questionSubId": 3425
# 30                            },
# 31                            "bbox": [[1,2,3,4]],
# 32                            "blanks": [
# 33                                {
# 34                                    "bbox": [[1,2,3,4]],
# 35                                    "segments": [
# 36                                        {
# 37                                            "content": ["x+1=2", "x+1=l", "x+1=3", "x+l=2", "c+l=2"],
# 38                                            "confidence": [0.99, 0.21, 0.10, 0.01, 0.001],
# 39                                            "bbox": [1,2,3,4]
# 40                                        }
# 41                                    ],
# 42                                    "selected_option": "A"
# 43                                }
# 44                            ],
# 45                            "typewritten": [
# 46                                {
# 47                                    "content": ["x+1=2", "x+1=l", "x+1=3", "x+l=2", "c+l=2"],
# 48                                    "confidence": [ 0.99, 0.21, 0.10, 0.01, 0.001],
# 49                                    "bbox": [1,2,3,4]
# 50                                }
# 51                            ],
# 52                            "sketches": [
# 53                                {
# 54                                    "content": ["x+1=2", "x+1=l", "x+1=3", "x+l=2", "c+l=2"],
# 55                                    "confidence": [0.99, 0.21, 0.10, 0.01, 0.001],
# 56                                    "bbox": [1,2,3,4]
# 57                                }
# 58                            ],
# 59                            "student-diagram": ["# the representation here is not decided yet. our current api does not return diagrams for /group endpoint so this is a flexible structure to support when we recognize aux lines inside diagrams or questions that require drawings"]
# 60                        }
# 61                    ]
# 62                }
# 63            ]
# 64        }
# 65    }
# 66}