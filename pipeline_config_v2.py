import os
from os.path import abspath, dirname
import logging
from importlib import reload
from enum import Enum

class PageMatchingMethods(Enum):
    CHARACTER_MATCHING=1
    TOKEN_MATCHING=2
    TF_IDF=3

class PipelineConfig:
    ROOT = abspath(dirname(__file__))
    CHECKPOINTS = {
        'sr': {
            'local_pth': os.path.join(ROOT, './checkpoints/sr_ckpt_1_net_G.pth'),
            'aws_pth': 'checkpoints/sr_ckpt_1_net_G.pth'},
        'sr_new': {
            'local_pth': os.path.join(ROOT, './checkpoints/sr_new_ckpt_1_net_G.pth'),
            'aws_pth': 'checkpoints/sr_new_ckpt_1_net_G.pth'},
        'size_detection': {
            'local_pth': os.path.join(ROOT, './checkpoints/size_det.pth.tar'),
            'aws_pth': 'checkpoints/size_det.pth.tar'},
        'valid_area_detection': {
            'local_pth': os.path.join(ROOT, './checkpoints/valid_area_detection_10092021.onnx'),
            'aws_pth': 'checkpoints/valid_area_detection_10092021.onnx'},
        'cn_essay_detection': {
            'local_pth': os.path.join(ROOT, './checkpoints/det_state_dict.pt'),
            'aws_pth': 'checkpoints/det_state_dict.pt'},
        'cn_essay_recog': {
            'local_pth': os.path.join(ROOT, './checkpoints/cn_recog_model.pth'),
            'aws_pth': 'checkpoints/cn_recog_model.pth'},
        'seg': {
            'local_pth': os.path.join(ROOT, './checkpoints/segmentation_pse_0620.pth'),
            'aws_pth': 'checkpoints/segmentation_pse_0620.pth'},
        'handwritten_seg': {
            'local_pth': os.path.join(ROOT, './checkpoints/handwritten_seg.mdl'),
            'aws_pth': 'checkpoints/handwritten_seg.mdl'},
        'graph_seg': {
            'local_pth': os.path.join(ROOT, './checkpoints/graph_seg_best_05212020.mdl'),
            'aws_pth': 'checkpoints/graph_seg_best_05212020.mdl'},
        'seg_fill_blank': {
            'local_pth': os.path.join(ROOT, './checkpoints/fib_seg_04302021.mdl'),
            'aws_pth': 'checkpoints/fib_seg_04302021.mdl'},
        'seg_high_school': {
            'local_pth': os.path.join(ROOT, './checkpoints/print_segment_0113_best.mdl'),
            'aws_pth': 'checkpoints/print_segment_0113_best.mdl'},
        'seg_exercise_book': {
            'local_pth': os.path.join(ROOT, './checkpoints/exercise_book_seg_02102021.mdl'),
            'aws_pth': 'checkpoints/exercise_book_seg_02102021.mdl'},
        'seg_exercise_book_53': {
            'local_pth': os.path.join(ROOT, './checkpoints/exercise_book_53_seg_11162020.mdl'),
            'aws_pth': 'checkpoints/exercise_book_53_seg_11162020.mdl'},
        'seg_elite_two_level_simeq': {
            'local_pth': os.path.join(ROOT, './checkpoints/elite_seg_grey_08262021.mdl'),
            'aws_pth': 'checkpoints/elite_seg_grey_08262021.mdl'},
        'seg_elite_chinese': {
            'local_pth': os.path.join(ROOT, './checkpoints/elite_chinese_seg_03102021.mdl'),
            'aws_pth': 'checkpoints/elite_chinese_seg_03102021.mdl'},
        'seg_elite_chem': {
            'local_pth': os.path.join(ROOT, './checkpoints/elite_chem_seg_03102021.mdl'),
            'aws_pth': 'checkpoints/elite_chem_seg_03102021.mdl'},
        'seg_elite_eng': {
            'local_pth': os.path.join(ROOT, './checkpoints/elite_eng_seg_03302021.mdl'),
            'aws_pth': 'checkpoints/elite_eng_seg_03302021.mdl'},
        'graph_seg_53': {
            'local_pth': os.path.join(ROOT, './checkpoints/graph_seg_53_11162020.mdl'),
            'aws_pth': 'checkpoints/graph_seg_53_11162020.mdl'},
        'graph_seg_elite': {
            'local_pth': os.path.join(ROOT, './checkpoints/graph_seg_elite_03042021.mdl'),
            'aws_pth': 'checkpoints/graph_seg_elite_03042021.mdl'},
        'graph_seg_elite_grey': {
            'local_pth': os.path.join(ROOT, './checkpoints/graph_seg_elite_gray_05182021.mdl'),
            'aws_pth': 'checkpoints/graph_seg_elite_gray_05182021.mdl'},
        'graph_seg_elite_grey_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/graph_seg_elite_gray_03142022.onnx'),
            'aws_pth': 'checkpoints/graph_seg_elite_gray_03142022.onnx'},
        'seg_graph_inner': {
            'local_pth': os.path.join(ROOT, './checkpoints/geosolver_08272021.mdl'),
            'aws_pth': 'checkpoints/geosolver_08272021.mdl'},
        'seg_english_46': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_english_46_09302021.mdl'),
            'aws_pth': 'checkpoints/seg_english_46_09302021.mdl'},
        'seg_one_eb_chn': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_chn_12132021.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_chn_12132021.mdl'},
        'seg_one_eb_handheld': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_handheld_10282021.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_handheld_10282021.mdl'},
        'seg_one_eb_eng_fib': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_eng_fib_08252022.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_eng_fib_08252022.mdl'},
        'seg_one_eb_phy_fib': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_phy_fib_12312021.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_phy_fib_12312021.mdl'},
        'seg_one_eb_math_fib': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_math_02022023.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_math_02022023.mdl'},
        'seg_one_eb_math_fib_onnx': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_math_fib_09232022.onnx'),
            'aws_pth': 'checkpoints/seg_one_eb_math_fib_09232022.onnx'},
        'seg_one_eb_eng_fib_onnx': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_eng_fib_08252022.onnx'),
            'aws_pth': 'checkpoints/seg_one_eb_eng_fib_08252022.onnx'},
        'seg_one_eb_eng_align': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_eng_01212023.mdl'),
            'aws_pth': 'checkpoints/seg_one_eb_eng_01212023.mdl'},
        'seg_one_eb_chn_fib_onnx': {
            'local_pth': os.path.join(ROOT, './checkpoints/seg_one_eb_chn_04152022.onnx'),
            'aws_pth': 'checkpoints/seg_one_eb_chn_04152022.onnx'},
        'blank_english':{
            'local_pth': os.path.join(ROOT, './checkpoints/blank_seg_english_08022022.mdl'),
            'aws_pth': 'checkpoints/blank_seg_english_08022022.mdl'},
        'blank_math':{
            'local_pth': os.path.join(ROOT, './checkpoints/blank_math_09052022.mdl'),
            'aws_pth': 'checkpoints/blank_math_09052022.mdl'},
        'blank_math_onnx':{
            'local_pth': os.path.join(ROOT, './checkpoints/blank_math_09052022.onnx'),
            'aws_pth': 'checkpoints/blank_math_09052022.onnx'},
        'seg_post': {},
        'math_transcription_hand_encoder_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_math_encoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_math_encoder_cpu_12222022.onnx'},
        'math_transcription_hand_embedding_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_math_embedding_cpu_12222022.pth'),
            'aws_pth': 'checkpoints/transcription_hand_math_embedding_cpu_12222022.pth'},
        'math_transcription_hand_decoder_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_math_decoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_math_decoder_cpu_12222022.onnx'},
        'english_transcription_hand_encoder_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_english_encoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_english_encoder_cpu_12222022.onnx'},
        'english_transcription_hand_embedding_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_english_embedding_cpu_12222022.pth'),
            'aws_pth': 'checkpoints/transcription_hand_english_embedding_cpu_12222022.pth'},
        'english_transcription_hand_decoder_cpu': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_english_decoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_english_decoder_cpu_12222022.onnx'},
        'exercise_book_transcription_hand_encoder': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_math_encoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_math_encoder_cpu_12222022.onnx'},
        'exercise_book_transcription_hand_decoder': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_math_embedding_cpu_12222022.pth'),
            'aws_pth': 'checkpoints/transcription_hand_math_embedding_cpu_12222022.pth'},
        'english_transcription_hand_encoder': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_english_encoder_cpu_12222022.onnx'),
            'aws_pth': 'checkpoints/transcription_hand_english_encoder_cpu_12222022.onnx'},
        'english_transcription_hand_decoder': {
            'local_pth': os.path.join(ROOT, './checkpoints/transcription_hand_english_embedding_cpu_12222022.pth'),
            'aws_pth': 'checkpoints/transcription_hand_english_embedding_cpu_12222022.pth'},
        'type_transcription_pipeline_split': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_split_06232021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_split_06232021.onnx'},
        'type_transcription_pipeline_ctc': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_ctc_flat_08052021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_ctc_flat_08052021.onnx'},
        'type_transcription_pipeline_ctc_3d': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_ctc_flat_math_08-08_version0.onnx'),
            'aws_pth': 'checkpoints/type_transcription_ctc_flat_math_08-08_version0.onnx'},
        'type_transcription_pipeline_ctc_ENG': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_ctc_flat_eng_09222021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_ctc_flat_eng_09222021.onnx'},
        'type_transcription_pipeline_ctc_ENG_3d': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_ctc_flat_eng_08-08_version0.onnx'),
            'aws_pth': 'checkpoints/type_transcription_ctc_flat_eng_08-08_version0.onnx'},
        'type_transcription_pipeline_ctc_chn': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_ctc_flat_chn_08052021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_ctc_flat_chn_08052021.onnx'},
        'type_transcription_pipeline_equation_encoder': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline_encoder_08052021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_multiline_encoder_08052021.onnx'},
        'type_transcription_pipeline_equation_encoder_3d': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline3d_encoder.onnx'),
            'aws_pth': 'checkpoints/type_transcription_multiline3d_encoder.onnx'},
        'type_transcription_pipeline_equation_decoder': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline_decoder_08052021.pth'),
            'aws_pth': 'checkpoints/type_transcription_multiline_decoder_08052021.pth'},
        'type_transcription_pipeline_equation_decoder_3d': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline_decoder_07152022.pth'),
            'aws_pth': 'checkpoints/type_transcription_multiline_decoder_07152022.pth'},
        'type_transcription_pipeline_equation_decoder_onnx': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline_decoder_08052021.onnx'),
            'aws_pth': 'checkpoints/type_transcription_multiline_decoder_08052021.onnx'},
        'type_transcription_pipeline_equation_decoder_onnx_3d': {
            'local_pth': os.path.join(ROOT, 'checkpoints/type_transcription_multiline3d_decoder.onnx'),
            'aws_pth': 'checkpoints/type_transcription_multiline3d_decoder.onnx'},
        'segordering': {
            'local_pth': os.path.join(ROOT, './checkpoints/segorder_classic_03152021.pth'),
            'aws_pth': 'checkpoints/segorder_classic_03152021.pth'},
        '53_segordering': {
            'local_pth': os.path.join(ROOT, './checkpoints/segorder_53_05202021.pth'),
            'aws_pth': 'checkpoints/segorder_53_05202021.pth'},
        'hand_segordering': {
            'local_pth': os.path.join(ROOT, './checkpoints/segorder_hand_03152021.pth'),
            'aws_pth': 'checkpoints/segorder_hand_03152021.pth'},
        'elite_segordering': {
            'local_pth': os.path.join(ROOT, './checkpoints/segorder_elite_03242021.pth'),
            'aws_pth': 'checkpoints/segorder_elite_03242021.pth'},
        'elite_onecol_segordering': {
            'local_pth': os.path.join(ROOT, './checkpoints/segorder_elite_onecol_03302021.pth'),
            'aws_pth': 'checkpoints/segorder_elite_onecol_03302021.pth'},
        'region_group_detection': {'local_pth': os.path.join(ROOT, './checkpoints/rg_det_03092021.onnx'),
                                   'aws_pth': 'checkpoints/rg_det_03092021.onnx'},
        'fill_blank_pair': {},
        'question_detection': {'local_pth': os.path.join(ROOT, './checkpoints/question_det_chn0712.onnx'),
                               'aws_pth': 'checkpoints/question_det_chn0712.onnx'},

        'question_detection_chem': {'local_pth': os.path.join(ROOT, './checkpoints/qd_chem0919_330.onnx'),
                'aws_pth': 'checkpoints/qd_chem0919_330.onnx'},
        'question_detection_math': {'local_pth': os.path.join(ROOT, './checkpoints/qd_math0922_750.onnx'),
                'aws_pth': 'checkpoints/qd_math0922_750.onnx'},
        'question_detection_chn': {'local_pth': os.path.join(ROOT, './checkpoints/chn1017_csv_retinanet_100.onnx'),
                'aws_pth': 'checkpoints/chn1017_csv_retinanet_100.onnx'},
        'question_detection_eng': {'local_pth': os.path.join(ROOT, './checkpoints/engrg_csv_retinanet_130.onnx'),
                'aws_pth': 'checkpoints/engrg_csv_retinanet_130.onnx'},
        'question_detection_eng2': {'local_pth': os.path.join(ROOT, './checkpoints/eng_qd2_02082022.onnx'),
                'aws_pth': 'checkpoints/eng_qd2_02082022.onnx'},
        'question_detection_math2': {'local_pth': os.path.join(ROOT, './checkpoints/qd2_v3_08142022_e3.onnx'),
                'aws_pth': 'checkpoints/qd2_v3_08142022_e3.onnx'},
        'question_detection_math3': {'local_pth': os.path.join(ROOT, './checkpoints/qd_math08182022.onnx'),
                'aws_pth': 'checkpoints/qd_math08182022.onnx'},
        'question_detection_eng3': {'local_pth': os.path.join(ROOT, './checkpoints/qd_eng08252022.onnx'),
                'aws_pth': 'checkpoints/qd_eng08252022.onnx'},
        'segment_grouping_eng': {'local_pth': os.path.join(ROOT, './checkpoints/segment_grouping_eng.ckpt'),
                'aws_pth': 'checkpoints/segment_grouping_eng.ckpt'},
        'segment_grouping_math': {'local_pth': os.path.join(ROOT, './checkpoints/segment_grouping_math.ckpt'),
                'aws_pth': 'checkpoints/segment_grouping_math.ckpt'},
        'img_align_kp_det': {'local_pth': os.path.join(ROOT, './checkpoints/img_align_kp_det.pth'),
                'aws_pth': 'checkpoints/img_align_kp_det.pth'},
        'img_align_kp_match': {'local_pth': os.path.join(ROOT, './checkpoints/img_align_kp_match.pth'),
                'aws_pth': 'checkpoints/img_align_kp_match.pth'},
        'img_align_kp_det_mymodel': {'local_pth': '/data/home/yanghanlong/workspace/SuperGlue-pytorch/logs/mymodel_v3_5_7/checkpoints/SuperPoint_epoch_14.pth'
        # 'img_align_kp_det_mymodel': {'local_pth': os.path.join(ROOT, './checkpoints/v2_3/SuperPoint_v2_3_epoch_28.pth')
                },
        'img_align_kp_match_mymodel': {'local_pth': '/data/home/yanghanlong/workspace/SuperGlue-pytorch/logs/mymodel_v3_5_7/checkpoints/SuperGlue_epoch_14.pth'
        # 'img_align_kp_match_mymodel': {'local_pth': os.path.join(ROOT, './checkpoints/v2_3/SuperGlue_v2_3_epoch_28.pth')
                },
        'mm_analysis_3head': {
            'local_pth': os.path.join(ROOT, './checkpoints/mm_3head_0120.pth'),
            'aws_pth': 'checkpoints/mm_3head_0120.pth',
        }
    }

    EVAL_DATA = {
        # 'eval_all_v2': {
        #     'local_pth': os.path.join(ROOT, './data/old_eval'),
        #     'aws_bucket': 'ocr-shared-storage',
        #     'aws_pth': 'processed/ocr_all_labeled_struct_v3_Jul2020/test'},
        'eval_all_v3': {
            'local_pth': os.path.join(ROOT, './data/evaluation'),
            'aws_bucket': 'ocr-shared-storage',
            'aws_pth': 'pipeline_data/evaluation'},
        'robustness': {
            'local_pth': os.path.join(ROOT, './data/robustness'),
            'aws_bucket': '',
            'aws_pth': 'pipeline_data/robustness'},
        'speed': {
            'local_pth': os.path.join(ROOT, './data/speed'),
            'aws_bucket': 'ocr-shared-storage',
            'aws_pth': 'pipeline_data/speed'},
        'prod_sample': {
            'local_pth': os.path.join(ROOT, './data/production_data_sample'),
            'aws_bucket': 'ocr-shared-storage',
            'aws_pth': 'pipeline_data/production_data_sample'},
        'cpu_speed': {
            'local_pth': os.path.join(ROOT, './data/meta_v2_math'),
            'aws_bucket': 'ocr-shared-storage',
            'aws_pth': 'pipeline_data/meta_v2_math'},

    }
    COMPILED_CHECKPOINT_DIR = os.path.join(ROOT, './trt_checkpoints/')
    LOG_DIR = os.path.join(ROOT, './logs/pipeline.log')  # deprecated
    TMP_DEBUG_RESULT_DIR = os.path.join(ROOT, './debug/')
    RUNTIME_RESULT_DIR = os.path.join(ROOT, "./logs/debug/")
    EVAL_DATA_DIR = os.path.join(ROOT, './evaluation/resource/')
    EVAL_RESULT_DIR = os.path.join(ROOT, './logs/results/')
    EVAL_BASELINE_DIR = os.path.join(ROOT, './evaluation/baselines/')
    TEST_DATA_DIR = os.path.join(ROOT, './test/resource/')
    TRANSCRIPT_N_CANDIDATES = 5  # should be a number between 1 and 5
    HAND_ENGLISH_LEXICON_PATH = os.path.join(ROOT, './libs/hand_transcript_src/data/english_chinese_lexicon.json')
    HAND_EVAL_LEXICON_PATH = os.path.join(ROOT, './libs/ops/hand_test_lexicon.json')
    TYPE_LEXICON_PATH = os.path.join(ROOT, './libs/ops/type_test_lexicon.json')
    ROBUSTNESS_TEST_DATA_DIR = os.path.join(ROOT, './data/robustness/')
    LOCAL_META_PATH = os.path.join(ROOT, "libs/region_grouping_src/meta/meta_detailed.json")
    LOCAL_META_V2_PATH = os.path.join(ROOT, "libs/structuralize_src/meta/sample_P14-15_newer.json")
    PAGE_MATCHING_METHOD = PageMatchingMethods.TOKEN_MATCHING
    # API
    QUEUE_IP = os.environ.get("MQ_HOST", "54.90.219.231")
    QUEUE_CONN_RETRY_LIMIT = 120
    QUEUE_PORT = 5672
    RETRY_REFRESH_TIME = 300
    RETRY_COUNT_LIMIT = 40
    RETRY_IMAGE_DOWNLOAD_LIMIT = 3
    MAX_SUCCESS_DURATION = 1800
    # CI and debugging
    GET_PROD_REQUEST_FROM_ID_URL = "http://data-processing.kezhitech.com/api/algo/ocr/findNewParamsByImageId?imageId={}"
    GET_PREPROD_REQUEST_FROM_ID_URL = "http://data-processing.kezhitech.com/api/algo/ocr/findNewParamsByImageId?imageId={}"
    GET_STAGE_REQUEST_FROM_ID_URL = "http://data-processing.kezhitech.com/api/algo/ocr/findNewParamsByImageId?imageId={}"




logging.shutdown()

if os.environ.get('ENV', '') in ['stage', 'preprod-dev', 'preprod-test', 'prod']:
    logging.basicConfig(format='%(asctime)-22s %(levelname)-8s - %(funcName)s - %(lineno)d - %(message)s',
                        level=logging.INFO)
else:
    if not os.path.exists(os.path.dirname(PipelineConfig.LOG_DIR)):
        os.makedirs(os.path.dirname(PipelineConfig.LOG_DIR))
    open(PipelineConfig.LOG_DIR, 'a').close()
    logging.basicConfig(format='%(levelname)-8s %(asctime)-22s - %(funcName)s - %(lineno)d - %(message)s',
                        filename=PipelineConfig.LOG_DIR,
                        level=logging.INFO)

# exercise book data endpoint
EB_BASE_URL = {'prod': 'http://api.qb.kezhitech.com/question-domain',
               'preprod-dev': 'http://8.140.146.94:10016',
               'preprod-test': 'http://8.140.146.94:10016',
               'stage': 'http://8.140.146.94:10016'}
EXERCISE_BOOK_QUESTIONS_INFO_ENDPOINT = '/bookPageQuestion/queryAllSubQuestionByBookId'

RABBIT_MQ_ENV_CONFIG = {
    'prod': {
        'username': 'ocr_pipeline',
        'broker_id': 'b-7055060b-cf2f-452a-ad48-22fe401236fd',
        'region': 'cn-north-1',
        'virtual_host': 'prod',
        'heartbeat': 0
    },
    'preprod-test': {
        'username': 'ocr_pipeline',
        'broker_id': 'b-5134fe4c-e4dc-4192-b40a-0970060f6e11',
        'region': 'us-east-1',
        'virtual_host': 'preprod-test',
        'heartbeat': 0
    },
    'preprod-dev': {
        'username': 'ocr_pipeline',
        'broker_id': 'b-5134fe4c-e4dc-4192-b40a-0970060f6e11',
        'region': 'us-east-1',
        'virtual_host': 'preprod-dev',
        'heartbeat': 0
    },
    'stage': {
        'username': 'ocr_pipeline',
        'broker_id': 'b-5134fe4c-e4dc-4192-b40a-0970060f6e11',
        'region': 'us-east-1',
        'virtual_host': 'stage',
        'heartbeat': 0
    }
}
PASSAGE_TYPES = [ 241,
                  242,
                  243,
                  244,
                  245,
                  247,
                  248,
                  249,
                  250,
                  251,
                  381,
                  382,
                  383,
                  384,
                  385,
                  387,
                  388,
                  389,        
                  390,
                  391]
