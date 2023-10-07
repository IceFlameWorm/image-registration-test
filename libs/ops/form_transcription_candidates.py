import numpy as np
from libs.util.misc_util import cvt_modelgrammar2outputgrammar


def ind2smart_inds(cand_ind):
    '''
    convert index to smart index for numpy

    :param cand_ind:candidate indices for current best sequence
    :return: smart index that can be used to get the sequence from one_line_segs
    '''
    cand_ind_expanded = cand_ind
    col_seg_ind = np.arange(0, len(cand_ind), dtype=np.int)
    return (col_seg_ind, cand_ind_expanded)


def form_candidates(preds, scores, n_best, index2token):
    '''
    form best transcription candidates with transcription candidates for each column segment

    :param preds: a list of candidate prediction out (n_col_segs, number of candidates for each column segment, length of predition results)
    :param scores: a list of scores for candidates (n_col_segs, number of candidates for each column segment)
    :param n_best: number of best sequences we want
    :param index2token: dict to transform index to tokens
    :return: best candidates, (n_best, n_candidates)
    '''

    preds_array = np.array(preds,dtype=object)
    scores_array = np.array(scores)

    n_col_seg, n_cand = scores_array.shape
    if n_best > n_cand:
        raise ValueError('when form top candidates, n_best cant be bigger than number of candidates on each col seg')

    best_cand_score = np.sum(scores_array, axis=1)
    cur_cand_score = np.zeros((best_cand_score.shape[0])) + best_cand_score
    cur_cand_ind = np.zeros((best_cand_score.shape[0]), dtype=np.int)

    best_candidates = [np.concatenate(preds_array[ind2smart_inds(cur_cand_ind)])]
    for i in range(1, n_best):
        prev_ind_score = scores_array[ind2smart_inds(cur_cand_ind)]
        cur_ind_score = scores_array[ind2smart_inds(cur_cand_ind+1)]
        step_cand_score = cur_cand_score - prev_ind_score + cur_ind_score

        chosen_ind = np.argmax(step_cand_score)
        cur_cand_ind[chosen_ind] += 1
        out_candidate_inds = np.zeros((best_cand_score.shape), dtype=np.int)
        out_candidate_inds[chosen_ind] += cur_cand_ind[chosen_ind]
        best_candidates.append(np.concatenate(preds_array[ind2smart_inds(out_candidate_inds)]))
    for i in range(len(best_candidates)):
        best_candidates[i] = ' '.join([index2token[int(ind)] for ind in best_candidates[i]])
        best_candidates[i] = cvt_modelgrammar2outputgrammar(best_candidates[i])

    # Filter out candidates with improper bracket counts. Note that this does not change scores.
    to_remove = []
    for idx, candidate in enumerate(best_candidates):
        num_open_brackets = 0
        frac_brackets_expected = [0]
        minimum_open_bracket_level = [0]
        bad_format = False
        for token in candidate.split():
            if token == '{':
                num_open_brackets += 1
            elif token == '}':
                if num_open_brackets < 1:
                    bad_format = True
                num_open_brackets -= 1
                if frac_brackets_expected[-1] > 0:
                    frac_brackets_expected[-1] -= 1
                    if frac_brackets_expected[-1] == 0:
                        frac_brackets_expected.pop()
                        minimum_open_bracket_level.pop()
            elif token == '\\frac':
                frac_brackets_expected.append(2)
                minimum_open_bracket_level.append(num_open_brackets)
            elif num_open_brackets <= minimum_open_bracket_level[-1] and frac_brackets_expected[-1] > 0:
                bad_format = True
        if num_open_brackets or frac_brackets_expected[-1] or bad_format:
            to_remove.append(idx)
    if len(to_remove) < len(best_candidates):
        replacement_idx = min([x for x in range(len(best_candidates)) if x not in to_remove])
        for idx in to_remove:
            best_candidates[idx] = best_candidates[replacement_idx]
    return best_candidates
