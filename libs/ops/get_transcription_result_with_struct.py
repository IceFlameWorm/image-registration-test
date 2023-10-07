
def get_trancription_result_with_struct(result_struct):
    '''

    Get transcription result from result struct, the result struct is defined on
    base_api, when we call the function, the result struct should already be filled
    by ocr pipeline

    Args:
        result_struct: Any
            The type of the result struct is too complex to write out. Check libs/base_api.py
            for the schema of this struct
    Returns:
        type_string: str
            result of typewritten transcription model
        hand_string: str
            result of handwritten transcription model
    '''

    hand_segs = result_struct['region']['handwritten']['segments']
    type_segs = result_struct['region']['printed']['segments']

    hand_results_list = []
    for seg in hand_segs:
        if seg['content']['hand'] is None:
            hand_results_list.append('[empty]')
        else:
            hand_results_list.append(seg['content']['hand'][0])

    type_results_list = []
    for seg in type_segs:
        if seg['content']['type'] is None:
            type_results_list.append('[empty]')
        else:
            type_results_list.append(seg['content']['type'][0])


    type_string = ' || '.join(type_results_list)
    hand_string = ' || '.join(hand_results_list)
    return type_string, hand_string


def get_trancription_result_lengths_with_struct(result_struct):
    '''

    Get transcription result from result struct, the result struct is defined on
    base_api, when we call the function, the result struct should already be filled
    by ocr pipeline

    Args:
        result_struct: Any
            The type of the result struct is too complex to write out. Check libs/base_api.py
            for the schema of this struct
    Returns:
        type_transcription_res_lengths: List[int]
            lengths for type transcription results for each line
        hand_transcription_res_lengths: List[int]
            lengths for hand transcription results for each line
    '''

    hand_segs = result_struct['region']['handwritten']['segments']
    type_segs = result_struct['region']['printed']['segments']

    hand_results_list = []
    for seg in hand_segs:
        if seg['content']['hand'] is None:
            hand_results_list.append('')
        else:
            hand_results_list.append(seg['content']['hand'][0])

    type_results_list = []
    for seg in type_segs:
        if seg['content']['type'] is None:
            type_results_list.append('')
        else:
            type_results_list.append(seg['content']['type'][0])

    hand_transcription_res_lengths = [str(len(hand_transcription_line)) for hand_transcription_line in hand_results_list]
    hand_transcription_res_lengths = ' '.join(hand_transcription_res_lengths)
    type_transcription_res_lengths = [str(len(type_transcription_line)) for type_transcription_line in type_results_list]
    type_transcription_res_lengths = ' '.join(type_transcription_res_lengths)
    return type_transcription_res_lengths, hand_transcription_res_lengths