import cv2
from libs.util.image_util import parse_svg
from libs.util.image_util import resize_svg
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np


def _draw_transcription_res(orig_im, svgs, transcriptions):
    '''

    draw transcription result given the background image
    svgs indicating location for the transcripted segments
    and the transcripted texts

    Args:
        orig_im: Uint8Ndarray
            input image to pipeline or the binarized image, must be gray scale
        svgs: List[str]:
            svgs representing contours
        transcriptions: List[str]
            transcription results

    Returns:
        transcription_res: Uint8Ndarray
            Image with transcription result on it. The background is the input image or binarized
            input image.

    '''
    file_dir_path = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(file_dir_path, 'simsun.ttc')
    h, w = orig_im.shape

    # set meta for drawing
    canvas = Image.fromarray(cv2.cvtColor(orig_im, cv2.COLOR_GRAY2BGR))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, 16)

    for i, svg in enumerate(svgs):
        right_scale_svg = resize_svg(svg, h, w)
        cnt = parse_svg(right_scale_svg)
        xy_left_up_coord = [min(cnt[:, 0]), min(cnt[:, 1])]  # its x, y coordinate
        if transcriptions[i] is None:
            transcription_res = 'empty'
        else:
            transcription_res = transcriptions[i][0]
        draw.text(xy_left_up_coord, transcription_res, font=font, fill=(0, 0, 255))
    transcription_res = np.array(canvas)
    return transcription_res


def get_transcription_ims(data):
    '''

    given result struct, get an image with transcription result on it.

    Args:
        data: Any
            The type of the structure for data is too complex to write out. Check libs/base_api.py
            for the schema of this struct

    Returns:
        type_transcription_res_im: Uint8Ndarray
            Image with type transcription result on it. The background is the input image or binarized
            input image.
        hand_transcription_res_im: Uint8Ndarray
            Image with hand transcription result on it. The background is the input image or binarized
            input image.
    '''

    type_segs = data['result_struct_ocr']['region']['printed']['segments']
    hand_segs = data['result_struct_ocr']['region']['handwritten']['segments']

    orig_im = data['img']['bin']

    type_svgs = [seg_struct['path'] for seg_struct in type_segs]
    type_transcriptions = [seg_struct['content']['type'] for seg_struct in type_segs]

    hand_svgs = [seg_struct['path'] for seg_struct in hand_segs]
    hand_transcriptions = [seg_struct['content']['hand'] for seg_struct in hand_segs]

    type_transcription_res_im = _draw_transcription_res(orig_im, type_svgs, type_transcriptions)
    hand_transcription_res_im = _draw_transcription_res(orig_im, hand_svgs, hand_transcriptions)

    return type_transcription_res_im, hand_transcription_res_im
