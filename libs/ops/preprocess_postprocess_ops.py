import pdb

# from libs.util.struct_util import struct_bbox_to_svg,add_relative_bbox_to_struct, bbox_mask_to_cnt, normalize_contour
from aiohttp import ClientSession
from aiobotocore.session import get_session
# from libs.ops.metrics.metrics import Metrics
from pipeline_config_v2 import EB_BASE_URL, EXERCISE_BOOK_QUESTIONS_INFO_ENDPOINT, PipelineConfig
import time
import logging
import os
import cv2
import numpy as np
import asyncio
import json
import re
from copy import deepcopy

from pipeline_config_v2 import PASSAGE_TYPES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


s3_session = get_session()
if os.environ.get('ENV', '') == 'prod':
    bucket_name = 'ocr-api-images-prod'
else:
    bucket_name = 'ocr-api-images-dev'

pipeline_config = PipelineConfig()

def _get_n_fib_line(subques_content):
    no_fib_line = subques_content.replace('\\_', '')
    return (len(subques_content) - len(no_fib_line)) // 2

def _get_subquestion_type(question_info, sub_question_info, n_blanks):
    if len(sub_question_info.get('option', [])) and n_blanks == 1:
        return 'MC'
    if sub_question_info['reviewQuestionType'] in PASSAGE_TYPES:
        return 'PASSAGE'
    test_text = sub_question_info['subQuestionBody']
    if (not len(test_text) or test_text == '.') and\
        not question_info is None:
        test_text = question_info['questionBody']
    if n_blanks == _get_n_fib_line(test_text):
        return 'FIB'
    return 'OTHER'

async def download_from_normal_url(url):
    start = time.time()
    count = 0
    image_data = None
    while count < pipeline_config.RETRY_IMAGE_DOWNLOAD_LIMIT:
        try:
            async with ClientSession() as session:
                response = await session.get(url, timeout=600)
                image_data = await response.read()
                duration = time.time() - start
                # Metrics.download_image_duration.add_duration(duration)
                logging.info('[Time] download user image from url use {}s'.format(round(duration, 4)))
                response.close()
                if image_data is None:
                    count += 1
                else:
                    break
        except asyncio.TimeoutError:
            logging.warning('download image from {} timeout!'.format(url))
            count += 1
        except Exception as e:
            logging.warning('download image from {} fail: {}'.format(url, e))
            count += 1
    if not image_data:
        raise ValueError('fail to get image data')
    return image_data


async def download_from_s3(url):
    # get object from s3
    start = time.time()
    count = 0
    while count < pipeline_config.RETRY_IMAGE_DOWNLOAD_LIMIT:
        try:
            async with s3_session.create_client('s3') as s3_client:
                response = await s3_client.get_object(Bucket=bucket_name, Key=url)
                # this will ensure the connection is correctly re-used/closed
                async with response['Body'] as stream:
                    image_data = await stream.read()
                    duration = time.time() - start
                    # Metrics.download_from_s3_duration.add_duration(duration)
                    logging.info('[Time] download user image from s3 use {}s'.format(round(duration, 4)))
                    if image_data is None:
                        count += 1
                    else:
                        break
        except asyncio.TimeoutError:
            logging.warning('download image from s3 {} timeout!'.format(url))
            count += 1
        except Exception as e:
            logging.warning('download image from s3 {} fail: {}'.format(url, e))
            count += 1
    if not image_data:
        raise ValueError('fail to get image data')
    return image_data


async def upload_s3(image, image_id, resize=False):
    # upload object to s3
    if not resize:
        key = f'preprocess/{image_id}.png'
    else:
        key = f'preprocess_resize/{image_id}.png'
    start = time.time()
    async with s3_session.create_client('s3') as s3_client:
        await s3_client.put_object(Bucket=bucket_name, Key=key, Body=image)
        duration = time.time() - start
        # Metrics.upload_to_s3_duration.add_duration(duration)
        logging.info('[Time] upload image to s3 use {}s'.format(round(duration, 4)))
    return key


async def download_img(url):
    if url.startswith('http'):
        img = await download_from_normal_url(url)
    else:
        img = await download_from_s3(url)
    return img

async def get_exercise_book_questions_info(data, env):
    EXERCISE_BOOK_QUESTIONS_INFO_URL = EB_BASE_URL[env] + EXERCISE_BOOK_QUESTIONS_INFO_ENDPOINT
    try:
        async with ClientSession() as session:
            res = await session.get(EXERCISE_BOOK_QUESTIONS_INFO_URL, params={'bookId': data['book_id']}, timeout=600)
            if res.status == 200:
                res = json.loads(await res.text())
                if res['code'] == 200:
                    meta = res['data']
                    logging.info(f"{data['img_key']} fetched meta data of exercise book {data['book_id']}({env})")
                    return meta
                else:
                    raise ValueError(f'call {EXERCISE_BOOK_QUESTIONS_INFO_URL} failed({env}), response: {res}')
            else:
                raise ValueError(f'call {EXERCISE_BOOK_QUESTIONS_INFO_URL} failed({env}), status code: {res.status}')
    except asyncio.TimeoutError:
        raise ValueError(f'call {EXERCISE_BOOK_QUESTIONS_INFO_URL} timeout({env})')
    except Exception as e:
        raise ValueError(f'call {EXERCISE_BOOK_QUESTIONS_INFO_URL} failed({env})')

def clean_online_meta(online_meta):
    # clean meta.questionSubList.subQuestionBody:    set to "" if equals "."
    if online_meta is None:
        return
    for question in online_meta:
        if "questionSubList" in question:
            for sub_question in question["questionSubList"]:
                if "subQuestionBody" in sub_question:
                    if sub_question["subQuestionBody"].strip() in ["."]:
                        sub_question["subQuestionBody"] = ""
    # set subquestion body in the senarios like:   question content: 36-41 and sub question body: 1-6
    for question in online_meta:
        # check if only one subqeustion exist and its subQuestionBody is numeric 
        if "questionSubList" in question and len(question["questionSubList"]) == 1 and question["questionSubList"][0]["subQuestionBody"].isnumeric():
            # get the start & end of displayQuestionNumber
            display_number = question["displayQuestionNumber"]
            pattern = re.compile(r"([0-9]+)+-([0-9]+)")
            match = pattern.match(display_number)
            if match is None:
                continue
            n_start = int(match.group(1))
            n_end = int(match.group(2))
            if n_start >= n_end:
                continue
            # set sub question body 
            sub_question_body = int(question["questionSubList"][0]["subQuestionBody"])
            # 16->25
            # 1/2/3/4/5/6...
            if sub_question_body < n_start:
                question["questionSubList"][0]["subQuestionBody"] = str(n_start + sub_question_body - 1)

def _merge_blocks(blocks):
    if len(blocks) <= 0:
        return blocks
    res = []
    merge_mode = 'dilate'
    if merge_mode == 'naive':
        block = blocks[0]
        min_x, min_y, max_x, max_y = block['rect']['startX'], block['rect']['startY'], block['rect']['endX'], block['rect']['endY']
        for block in blocks:
            x, y, w, h = block['rect']['startX'], block['rect']['startY'], block['rect']['width'], block['rect']['height']
            min_x = min(min_x, x)
            max_x = max(max_x, (x+w))
            min_y = min(min_y, y)
            max_y = max(max_y, (y+h))
        res = [{'type': None, 'rect': {'startX': min_x, 'startY': min_y, 'endX': max_x, 'endY': max_y, 'width': (max_x-min_x), 'height': (max_y-min_y), 'index': 1}}]
    elif merge_mode == 'dilate':
        # dilate boxes
        boxes = []
        for block in blocks:
            x, y, w, h = block['rect']['startX'], block['rect']['startY'], block['rect']['width'], block['rect']['height']
            boxes.append([x, y, w, h])
        # merge
        expand_times = 5
        for i in range(0, len(boxes)):
            box = boxes[i]
            if not box:
                continue
            for j in range(i+1, len(boxes)):
                ahead_box = boxes[j]
                if not ahead_box:
                    continue
                cmp_box = box[:]
                cmp_box[3] *= expand_times
                overlap_h = min(cmp_box[1]+cmp_box[3], ahead_box[1]+ahead_box[3]) - max(cmp_box[1], ahead_box[1])
                if overlap_h > 0:
                    # do merge
                    min_x = min(box[0], ahead_box[0])
                    min_y = min(box[1], ahead_box[1])
                    max_x = max(box[0] + box[2], ahead_box[0] + ahead_box[2])
                    max_y = max(box[1] + box[3], ahead_box[1] + ahead_box[3])
                    box = [min_x, min_y, max_x - min_x, max_y - min_y]
                    boxes[i] = box
                    boxes[j] = []
                else:
                    break
        res = []
        for i, box in enumerate(boxes):
            if not box:
                continue
            res.append({'type': None, 'rect': {'startX': box[0], 'startY': box[1], 'endX': box[0]+box[2], 'endY': box[1]+box[3], 'width': box[2], 'height': box[3], 'index': i+1}})
    else:
        assert False
    return res

async def preprocess_web_data(data, env):
    #  transform web raw data from [API] to pipeline input data structure
    try:
        img_key = data['img_key']
        img = await download_img(data['img_url'])
        img_np = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img_np, 0)
        img_color = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        orig_size = img.shape[:2]
    except:
        raise ValueError('decode image failed!')
    if 'img_size' in data:
        orig_size = data['img_size']
    valid_region = [0, 0, orig_size[1], orig_size[0]]  # [x1,y1,x2,y2]
    whole_page_valid_region = False  # if input image is whole page scale + valid region, therefore autosize is not needed

    # valid region
    if 'valid_region' in data:
        if data['valid_region']:
            if len(data['valid_region']) == 4:
                valid_region = data['valid_region']
                whole_page_valid_region = True
    coords = []
    student_id = {}
    # online meta
    use_online_meta = False
    online_meta = None
    if 'meta' in data:
        if data['meta']:
            use_online_meta = True
            online_meta = data['meta']
    elif 'question_info' in data:
        online_meta = data['question_info']
        use_online_meta = True
    elif 'book_id' in data:
        online_meta = await get_exercise_book_questions_info(data, env)
        use_online_meta = True

    quality = data.get('quality', 0)
    script_type = data.get('script_type', 0)
    question_position_info = data.get('question_position_info', []) or []
    # fix order is null
    # fix bug for: 7f66106c-bcba-45c9-891e-9a8c356c930c
    for i in range(len(question_position_info)):
        page_info = question_position_info[i]
        for j in range(len(page_info['questionList'])):
            question_info = page_info['questionList'][j]
            for k in range(len(question_info['subQuestionList'])):
                subquestion_info = question_info['subQuestionList'][k]
                if len(subquestion_info['blankList']) == 1 and subquestion_info['blankList'][0]['order'] is None:
                    logging.info(f'set order from null to 1. subquestion_info: {subquestion_info}')
                    subquestion_info['blankList'][0]['order'] = 1
                # fix bug for 793d343f-7ef6-4a71-9013-4850bb0e1a69
                subquestion_info['blankList'] = list(filter(lambda x: x['order'] is not None, subquestion_info['blankList']))
                question_info['subQuestionList'][k] = subquestion_info
            page_info['questionList'][j] = question_info
        question_position_info[i] = page_info
    
    # fix bug for: 008ff934-a96c-4a12-b170-49d16a04c144
    for i in range(len(question_position_info)):
        page_info = question_position_info[i]
        for j in range(len(page_info['questionList'])):
            question_info = page_info['questionList'][j]
            subq_used = 0
            for k in range(len(question_info['subQuestionList'])):
                subquestion_info = question_info['subQuestionList'][k]
                if not subquestion_info['blocks'] and not subquestion_info['blankList']:
                    # del it from page
                    logging.info(f'delete subquestion from page. subquestion_info: {subquestion_info}')
                    continue
                question_info['subQuestionList'][subq_used] = subquestion_info
                subq_used += 1
            question_info['subQuestionList'] = question_info['subQuestionList'][:subq_used]
            page_info['questionList'][j] = question_info
        question_position_info[i] = page_info

    # merge blank region for PASSAGE
    question_infos = data.get('question_info', []) or []
    passage_subq_ids = []
    for question_info in question_infos:
        for subquestion_info in question_info.get('questionSubList', []):
            if _get_subquestion_type(question_info, subquestion_info, subquestion_info['blankCount']) == 'PASSAGE':
                passage_subq_ids.append(subquestion_info['bookPageQuestionId'])
    for i in range(len(question_position_info)):
        page_info = question_position_info[i]
        for j in range(len(page_info.get('questionList', []))):
            question_info = page_info['questionList'][j]
            for k in range(len(question_info.get('subQuestionList', []))):
                subquestion_info = question_info['subQuestionList'][k]
                if subquestion_info['bookPageQuestionId'] in passage_subq_ids:
                    # merge blocks for blank
                    for blank in subquestion_info['blankList']:
                        blank['blocks'] = _merge_blocks(blank['blocks'])

    # ===================================== FLAG TRANSLATION LOGIC ===================================== #
    # MQ mode v2
    if 'subject' in data and 'blueprint_flag' not in data:
        subject = data['subject']
        flag = 'one_eb_rg2'
        student_id = data.get('student_info', {})
        image_type = data.get('image_type', 0)
        script_type = data.get('script_type', 0)

        if image_type is None:
            image_type = 0
        if image_type <= 0:
            if subject in [101, 102]:
                flag += '_math'
            elif subject in [402]:
                flag += '_phy'
            elif subject in [502]:
                flag += '_chem'
            elif subject in [301, 302]:
                flag += '_eng'
            elif subject in [201, 202, 702, 802, 902]:
                flag += '_chn'
            else:
                flag += '_math'
            
            if len(question_position_info) > 0:
                flag += '_align'

            if image_type == -1:
                flag += '_init'
        elif image_type == 1:
            flag += '_student_no'

    # web api v2, MQ mode v1
    elif 'blueprint_flag' in data:
        flag = data['blueprint_flag']
        if 'human_assist' in flag:
            coords = data['flags']['segments']
        elif flag.startswith('one_eb_region_grouped_student_id'):
            student_id = data.get('student_info', {})
        elif flag == 'one_eb_region_grouped_online_math' and quality == 1:
            flag += '_handheld'

    # web api v1
    else:
        subject = data['flags']['subject']  # 1: Math, 5: Chinese
        question_type = data['flags']['question_type']
        transcription_type = data['flags']['type']  # 1: Hand only, 2: Print only, 3: Both
        process_type = data['flags']['process_type']  # 0: one region, 1: preprocess only, 2: one page
        coords = data['flags']['segments']  # [[x1, y1, x2, y2], ...]
        exercise_book_type = 0
        if 'exercise_book_type' in data['flags']:
            exercise_book_type = data['flags']['exercise_book_type']

        flag = 'one_eb_autosize'  # default flag

        # ============ MATH ============
        # --- key points human assisted ---
        if len(coords) > 0:
            if transcription_type in [1, 3]:
                flag = 'math_human_assisted_segmentation_handwritten'
            if transcription_type == 2:
                flag = 'math_human_assisted_segmentation_printed'


        # === answersheets === #

        # Q&A question in answer sheets / question body
        elif question_type == 0:
            if quality == 1:
                flag = 'handheld_math_answersheet'
            elif quality == 0:
                if transcription_type in [1, 3]:  # answersheets
                    flag = 'scan_math_answersheet'
                if transcription_type == 2:  # question body
                    flag = 'scan_math_print_only'


        # fill in blank in answer sheets
        elif question_type == 1:
            flag = 'scan_math_answersheet_fill_in_blank'


        # whole page answersheets
        elif question_type == 2:
            pass  # not supported


        # === exercise books === #

        # exercise book, Q&A question
        elif question_type == 3:
            if whole_page_valid_region:
                flag = 'one_eb'
            else:
                flag = 'one_eb_autosize'
            if subject == 1:
                flag += '_math'
            elif subject == 2:
                flag += '_phy'
            elif subject == 3:
                flag += '_chem'
            elif subject == 4:
                flag += '_eng'
            elif subject == 5:
                flag += '_chn'

        # exercise book, whole page
        elif question_type == 4:
            if process_type == 0:  # no region
                if exercise_book_type == 0:
                    flag = 'one_eb'
                    if subject == 1:
                        flag += '_math'
                    elif subject == 2:
                        flag += '_phy'
                    elif subject == 3:
                        flag += '_chem'
                    elif subject == 4:
                        flag += '_eng'
                    elif subject == 5:
                        flag += '_chn'
                elif exercise_book_type == 1:
                    flag = '53'
                elif exercise_book_type == 2:
                    if subject in [1, 2]:
                        flag = 'elite_math_phy'
                    elif subject in [4, 7, 8, 9]:
                        flag = 'elite_eng'
                    elif subject in [5]:
                        flag = 'elite_chn'
                    elif subject in [3, 6]:
                        flag = 'elite_chem'
                    if transcription_type in [1, 3]:
                        flag += '_with_hand'
            elif process_type == 2:  # region grouped
                if use_online_meta:
                    flag = 'one_eb_region_grouped_online'
                    student_id = data.get('student_info', {})
                    if student_id:
                        flag = 'one_eb_region_grouped_student_id'
                    if subject == 1:
                        flag += '_math'
                        if quality == 1 and flag.startswith('one_eb_region_grouped_online'):
                            flag += '_handheld'
                    elif subject == 2:
                        flag += '_phy'
                    elif subject == 3:
                        flag += '_chem'
                    elif subject == 4:
                        flag += '_eng'
                    elif subject == 5:
                        flag += '_chn'
                else:
                    flag = 'one_eb_region_grouped_local'

            elif process_type == 3:  # graph only
                flag = 'elite_graph_only'


        # exercise book, fill in blank
        elif question_type == 5:
            if whole_page_valid_region:
                flag = 'one_eb_fill_in_blank'
            else:
                flag = 'one_eb_fill_in_blank_autosize'
            if subject == 1:
                flag += '_math'
            elif subject == 2:
                flag += '_phy'
            elif subject == 3:
                flag += '_chem'
            elif subject == 4:
                flag += '_eng'
            elif subject == 5:
                flag += '_chn'

        #  exercise book, calculation
        elif question_type == 6:
            if whole_page_valid_region:
                flag = 'one_eb'
            else:
                flag = 'one_eb_autosize'
            if subject == 1:
                flag += '_math'
            elif subject == 2:
                flag += '_phy'
            elif subject == 3:
                flag += '_chem'
            elif subject == 4:
                flag += '_eng'
            elif subject == 5:
                flag += '_chn'

        # # ============ CHINESE ============
        # if subject == 5:
        #     # === chinese essay ===
        #     if question_type == 0:
        #         flag = 'scan_chinese_essay'

        # ============ preprocess only ============
        if process_type == 1:
            flag = '*' + flag

        # ============ already preprocessed ============ [deprecated]
        if quality == 2:
            flag = '_' + flag
    
    clean_online_meta(online_meta)

    # 可能OOM 对img_color做些处理 短边缩放到1000
    target_pixels = min(1000,  min(img_color.shape[:2]))
    color_scale = target_pixels / min(img_color.shape[:2]) # <=1
    img_color = cv2.resize(img_color, (0,0), fx=color_scale, fy=color_scale, interpolation=cv2.INTER_AREA)

    data = {'id': img_key,
            'format': 'input',
            'info': {'image': {'orig': img,
                               'orig_color': img_color, 
                               'color_scale': color_scale,
                               'orig_size': orig_size,
                               'valid_region': valid_region,
                               'labeled_bboxs': coords
                               },
                     'meta': online_meta,
                     'student_id': student_id,
                     'script_type': script_type,
                     'question_position_info': question_position_info
                     }
            }
    return data, flag

async def posprocess_web_data(data, flag=''):
    if 'v2_struct' in data['info']:
        raw_output = {
            'imageId': data['id'],
            'status': 3,
            'status_message': 'success'
        }
        raw_output.update(data['info']['v2_struct'])

    else:
        raw_output = {'id': data['id'],
                    'image': {},
                    'struct': None,
                    'status': 3,
                    'status_message': 'success'}
        if 'orig_size' in data['info']['image']:
            h, w = data['info']['image']['orig_size']
        else:
            h, w = data['info']['image']['orig'].shape[:2]
        if 'struct' in data['info']:
            struct = data['info']['struct']
            raw_output['struct'] = struct_bbox_to_svg(struct, h, w)

    if flag.startswith('*'):
        processed_img = data['info']['image']['processed']
        processed_img_resized = cv2.resize(processed_img.copy(), (w, h))
        # processed_img = cv2.imencode('.png', processed_img)[1].tobytes()
        processed_img_resized = cv2.imencode('.png', processed_img_resized)[1].tobytes()
        image_key = await upload_s3(processed_img_resized, data['id'], resize=True)
        raw_output['image']['processed'] = image_key

    return raw_output



def dummy_posprocess_web_data(data, flag=''):
    """
    an regular copy of postprocess code to get debug result,
    PLEASE UPDATE THIS WHEN POSTPROESS IS UPDATED
    Args:
        data:
        flag:

    Returns:

    """

    raw_output = {'id': data['id'],
                  'image': {},
                  'struct': None,
                  'status': 3,
                  'status_message': 'success'}
    if 'orig_size' in data['info']['image']:
        h, w = data['info']['image']['orig_size']
    else:
        h, w = data['info']['image']['orig'].shape[:2]
    if 'struct' in data['info']:
        struct = data['info']['struct']
        raw_output['struct'] = struct_bbox_to_svg(struct, h, w)

    return raw_output


def join_bboxs(bboxs):
    if len(bboxs) == 0:
        return []
    bboxs=np.array(bboxs.copy())
    return [[int(np.min(bboxs[:,0])),
             int(np.min(bboxs[:,1])),
             int(np.max(bboxs[:,2])),
             int(np.max(bboxs[:,3]))]] #todo add 2 column support

def clip_bbox(bbox, pw, ph):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(pw-1, x0))
    x1 = max(0, min(pw-1, x1))
    y0 = max(0, min(ph-1, y0))
    y1 = max(0, min(ph-1, y1))
    return [x0, y0, x1, y1]

def ocr_api2_internal_to_external(struct,orig_h,orig_w,is_init_pipeline):
    """transform internal ocr api schema to external

    Args:
        struct ([type]): [description]
        orig_h ([type]): [description]
        orig_w ([type]): [description]

    Returns:
        [type]: [description]
    """    """
    """
    struct = deepcopy(struct)
    struct = add_relative_bbox_to_struct(struct)
    page_matched = struct.get("matched",False)
    output = {"page":{
        "pageNo": struct.get("page_no",-1),
        "matched": page_matched,
        "punchholeBoundingBoxList": struct.get("punchhole_bbox",[]),
        
        "studentInfo": {
            "studentId": struct['student_info']['student_id'] if 'student_info' in struct else '',
            "studentIdBoundingBox": struct['student_info']['student_id_bbox'] if 'student_info' in struct else [],
            "studentIdCandidateInfo": struct['student_info']['student_candidate_info'] if 'student_info' in struct else [],
            "studentName": '',
            "studentNameBoundingBox": struct['student_info']['student_name_bbox'] if 'student_info' in struct else []
        },
        "completionTime": struct.get("completion_time",-1),
        "questionList": struct.get("questions", [])
        }}
    
    
    
    for question in output["page"]["questionList"]:
        question_bboxs = []
        for subquestion in question["questionSubList"]:
            subquestion_bboxs = []
            # convert struct
            subquestion_clusters = convert_internal_subquestion(struct['child'][subquestion['region_id']],orig_h,orig_w, is_init_pipeline)
            # deal with blanks
            subquestion["blankList"] = []
            for blank_idx,blank in enumerate(subquestion_clusters["blanks"]):
                # blank_bbox = join_bboxs([segment['boundingBox'] for segment in blank])
                blank_bbox_mask = subquestion_clusters['blanks_extra_info'][blank_idx]['bbox_mask']
                blank_bbox = [[int(blank_bbox_mask['bbox']['x1']/blank_bbox_mask['canvas_size']['w']*orig_w),
                               int(blank_bbox_mask['bbox']['y1']/blank_bbox_mask['canvas_size']['h']*orig_h),
                               int(blank_bbox_mask['bbox']['x2']/blank_bbox_mask['canvas_size']['w']*orig_w),
                               int(blank_bbox_mask['bbox']['y2']/blank_bbox_mask['canvas_size']['h']*orig_h)]]
                blank_bbox = [clip_bbox(bbox, orig_w, orig_h) for bbox in blank_bbox]
                subquestion["blankList"].append(
                    {"boundingBoxList":blank_bbox, #  
                     "segmentList":blank,
                     "selectedOptionList": subquestion_clusters['blanks_extra_info'][blank_idx]['selectedOptionList'],
                     "extendInfo": {
                        "trustworthy": subquestion['blank_trustworthy_list'][blank_idx],
                        'uid': subquestion['blank_union_answer_uids'][blank_idx]['uid'] if 'blank_union_answer_uids' in subquestion else '',
                        'union_answer_uids': subquestion['blank_union_answer_uids'][blank_idx]['union_answer_uids']  if 'blank_union_answer_uids' in subquestion else [],
                        'candidate_segments': subquestion_clusters['blank_candidate_segments'][blank_idx]
                     }
                    })
            if 'blank_trustworthy_list' in subquestion:
                del subquestion['blank_trustworthy_list']
            if 'blank_union_answer_uids' in subquestion:
                del subquestion['blank_union_answer_uids']

            subquestion["sketchList"] = []
            if "sketches" in subquestion_clusters:
                if len(subquestion_clusters["sketches"]) > 0:
                    subquestion["sketchList"] = subquestion_clusters["sketches"][0].copy()

            subquestion["typewrittenList"] = [] 
            if "printed" in subquestion_clusters and len(subquestion_clusters["printed"]) > 0:
                subquestion["typewrittenList"] = subquestion_clusters["printed"][0].copy()

            
            # bbox
            # change keys:
            if subquestion["meta"] is None:
                subquestion["meta"] = {"matched": False,
                                       "similarity": 0.,
                                       "bookPageQuestionId": -1
                                      }
            else:
                subquestion["meta"]["matched"] = page_matched and subquestion["meta"]["matched"]

            if "question_sub_id" in subquestion["meta"]:
                subquestion["meta"]["bookPageQuestionId"] = subquestion["meta"]["question_sub_id"]
                del subquestion["meta"]["question_sub_id"]
            if "questionSubId" in subquestion["meta"]:
                subquestion["meta"]["bookPageQuestionId"] = subquestion["meta"]["questionSubId"]
                del subquestion["meta"]["questionSubId"]
            if "questionId" in subquestion["meta"]: # LM error
                subquestion["meta"]["bookPageQuestionId"] = subquestion["meta"]["questionId"]
                del subquestion["meta"]["questionId"]
            if "question_id" in subquestion["meta"]: # LM error
                subquestion["meta"]["bookPageQuestionId"] = subquestion["meta"]["question_id"]
                del subquestion["meta"]["question_id"]
            # delete all extra keys in meta
            allowed_keys = ["matched", "similarity", "bookPageQuestionId", "question_type"]
            all_subquestion_meta_keys = subquestion["meta"].keys()
            for key in list(all_subquestion_meta_keys).copy():
                if key not in allowed_keys:
                    del subquestion["meta"][key]

            # add bbox
            
            if struct['child'][subquestion['region_id']]['bbox']:
                subquestion['boundingBoxList'] = [[int(struct['child'][subquestion['region_id']]['bbox'][0]*orig_w),
                                                int(struct['child'][subquestion['region_id']]['bbox'][1]*orig_h),
                                                int(struct['child'][subquestion['region_id']]['bbox'][2]*orig_w),
                                                int(struct['child'][subquestion['region_id']]['bbox'][3]*orig_h)]]
                subquestion['boundingBoxList'] = [clip_bbox(bbox, orig_w, orig_h) for bbox in subquestion['boundingBoxList']]
            else:
                subquestion['boundingBoxList'] = []
            subquestion["studentDiagramList"] = []

            subquestion["extendInfo"] = {}
            subquestion["extendInfo"]["trustworthy"] = subquestion["trustworthy"]
            del subquestion["trustworthy"]

            question_bboxs+=subquestion['boundingBoxList']
            if 'region_id' in subquestion:
                del subquestion['region_id'] 
        
 

        
        if question["meta"] is None:
            question["meta"] = {"matched": False,
                                "similarity": 0.,
                                "questionId": -1
                               }
        else:
            question["meta"]["matched"] = page_matched and question["meta"]["matched"]
                    # delete all extra keys in meta
        allowed_keys = ["matched", "similarity", "questionId"]
        all_question_meta_keys = question["meta"].keys()
        for key in list(all_question_meta_keys).copy():
            if key not in allowed_keys:
                del question["meta"][key]
        # change keys:

        
        if struct['child'][question['region_id']]['bbox']:
            question['boundingBoxList'] = [[int(struct['child'][question['region_id']]['bbox'][0]*orig_w),
                                            int(struct['child'][question['region_id']]['bbox'][1]*orig_h),
                                            int(struct['child'][question['region_id']]['bbox'][2]*orig_w),
                                            int(struct['child'][question['region_id']]['bbox'][3]*orig_h)]]
            question['boundingBoxList'] = [clip_bbox(bbox, orig_w, orig_h) for bbox in question['boundingBoxList']]
        else:
            question['boundingBoxList'] = []
        if 'region_id' in question:
            del question['region_id'] 
    
    # clean dirty types
    if type(output["page"]["completionTime"]) != int:
        output["page"]["completionTime"] = -1
    if is_init_pipeline:
        output["page"]['NotMatchedHandSegments'] = []
        output["page"]['NotMatchedTypeSegments'] = []
        if struct.get('child', [{'type': 'placeholder'}])[-1]['type'] == 'not_grouped':
            notgrouped_region = struct['child'][-1]
            # segments in question but not in subquestions
            question_segments = {}
            for region in struct['child']:
                if region['type'] == 'question':
                    for cluster in region['child']:
                        if cluster['type'] in ('printed', 'blanks', 'sketches'):
                            for segment in cluster['child']:
                                question_segments[segment['id']] = segment
            subquestion_segments = {}
            for region in struct['child']:
                if region['type'] == 'subquestion':
                    for cluster in region['child']:
                        if cluster['type'] in ('printed', 'blanks', 'sketches'):
                            for segment in cluster['child']:
                                subquestion_segments[segment['id']] = segment
            left_segments = []
            for segment_id, segment in question_segments.items():
                if segment_id not in subquestion_segments:
                    left_segments.append(segment)
            for segment_idx, segment in enumerate(left_segments):
                for cluster in notgrouped_region['child']:
                    if cluster['type'] == 'sketches' and segment['type'] == 'handwritten':
                        cluster['child'].append(segment)
                    elif cluster['type'] == 'type' and segment['type'] == 'printed':
                        cluster['child'].append(segment)
                
            for cluster in notgrouped_region['child']:
                if cluster['type'] == 'sketches':
                    extra_h_segs = []
                    for segment_idx, segment in enumerate(cluster['child']):
                        converted_seg = convert_segment(segment_idx, segment, orig_h, orig_w, is_init_pipeline)
                        if not converted_seg is None:
                            extra_h_segs.append(converted_seg)
                    output["page"]['NotMatchedHandSegments'] = extra_h_segs
                if cluster['type'] == 'type':
                    extra_t_segs = []
                    for segment_idx, segment in enumerate(cluster['child']):
                        converted_seg = convert_segment(segment_idx, segment, orig_h, orig_w, is_init_pipeline)
                        if not converted_seg is None:
                            extra_t_segs.append(converted_seg)
                    output["page"]['NotMatchedTypeSegments'] = extra_t_segs
            
    return output

def convert_segment(segment_idx, segment, orig_h, orig_w, is_init_pipeline):
    if segment["bbox_mask"] is None:
        return None

    if is_init_pipeline:
        pos_info = segment.get('pos_info', None)
        bbox_mask = segment["bbox_mask"]
        if pos_info is not None:
            cnt_points = pos_info['relative']['contour']
        elif bbox_mask is not None:
            h, w = bbox_mask['canvas_size']['h'], bbox_mask['canvas_size']['w']
            cnt = bbox_mask_to_cnt(deepcopy(bbox_mask))
            cnt_points = normalize_contour(cnt,h,w)
        else:
            cnt_points = []
        segment_external = {"contentList": segment["content"],
                            "confidenceList": segment.get("content_confidences",[1.,1.,1.,1.,1.]),
                            "stepConfidences": segment.get("step_confidences", []),
                            "stepTokens": segment.get("step_tokens", []),
                            "targetContent": segment.get("target_content", []),
                            "targetConfidences": segment.get("target_confidences", []),
                            "targetStepConfidences": segment.get("target_step_confidences", []),
                            "targetStepTokens": segment.get("target_step_tokens", []),
                            "boundingBox": [int(segment["bbox_mask"]["bbox"]["x1"]/segment["bbox_mask"]["canvas_size"]["w"]*orig_w),
                                    int(segment["bbox_mask"]["bbox"]["y1"]/segment["bbox_mask"]["canvas_size"]["h"]*orig_h),
                                    int(segment["bbox_mask"]["bbox"]["x2"]/segment["bbox_mask"]["canvas_size"]["w"]*orig_w),
                                    int(segment["bbox_mask"]["bbox"]["y2"]/segment["bbox_mask"]["canvas_size"]["h"]*orig_h)],
                            # todo update bbox calcuate
                            "contourPoints": cnt_points,
                            "idx": segment_idx #segment.get("id",0)
                        }
    else:
        segment_external = {"contentList": segment["content"],
                            "confidenceList": segment.get("content_confidences",[1.,1.,1.,1.,1.]),
                            "stepConfidences": segment.get("step_confidences", []),
                            "stepTokens": segment.get("step_tokens", []),
                            "targetContent": segment.get("target_content", []),
                            "targetConfidences": segment.get("target_confidences", []),
                            "targetStepConfidences": segment.get("target_step_confidences", []),
                            "targetStepTokens": segment.get("target_step_tokens", []),
                            "boundingBox": [int(segment["bbox_mask"]["bbox"]["x1"]/segment["bbox_mask"]["canvas_size"]["w"]*orig_w),
                                    int(segment["bbox_mask"]["bbox"]["y1"]/segment["bbox_mask"]["canvas_size"]["h"]*orig_h),
                                    int(segment["bbox_mask"]["bbox"]["x2"]/segment["bbox_mask"]["canvas_size"]["w"]*orig_w),
                                    int(segment["bbox_mask"]["bbox"]["y2"]/segment["bbox_mask"]["canvas_size"]["h"]*orig_h)],
                            # todo update bbox calcuate
                            "idx": segment_idx #segment.get("id",0)
                        }
    processed_contents = []
    for content in segment_external["contentList"]:
        content = content.replace(" ","")
        # content = content.replace("\\tab"," ")
        processed_contents.append(content)
    segment_external["contentList"] = processed_contents
    return segment_external

def convert_internal_subquestion(region,orig_h,orig_w,is_init_pipeline):
    # region to question infomation
    
    result_question_info = {"printed": [], # list of external segments, 
                            "blanks": [],
                            "sketches": [],
                            "blanks_extra_info": [], # list of dictionaries, extra keys for blanks, len of this list should be the same as blanks
                            "blank_candidate_segments": []
                            }
    for cluster in region['child']: # iterate internal cluster
        converted_cluster = []
        # convert all segments into external segments
        for segment_idx, segment in enumerate(cluster['child']):
            # convert
            segment_external = convert_segment(segment_idx, segment, orig_h, orig_w, is_init_pipeline)
            # add converted segments into temp list of segments
            if not segment_external is None:
                converted_cluster.append(segment_external)
        result_question_info[cluster['type']].append(converted_cluster)
        if cluster['type'] == "blanks":
            result_question_info["blanks_extra_info"].append({'selectedOptionList': cluster["choice_option"],
                                                              'bbox_mask': cluster["bbox_mask"],
            })
            c_seg_ext_list = []
            for c_seg_idx, c_seg in enumerate(cluster.get('candidate_child', [])):
                c_seg_ext = convert_segment(c_seg_idx, c_seg, orig_h, orig_w, is_init_pipeline)
                if c_seg_ext is not None:
                    c_seg_ext_list.append(c_seg_ext)

            result_question_info["blank_candidate_segments"].append(c_seg_ext_list)

        if len(result_question_info["blanks"]) != len(result_question_info["blanks_extra_info"]):
            raise ValueError("some blanks have no choice option key")
            
    return result_question_info
