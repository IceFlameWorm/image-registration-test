import json
from pipeline_config_v2 import PipelineConfig

class Tokenizer:
    def __init__(self, lexicon_path):
        with open(lexicon_path, "r") as file:
            self.nl_tokens = json.load(file)
            self.token2index = {}
            for i, token in enumerate(self.nl_tokens):
                self.token2index[token] = i
            self.unk_ind = self.token2index['<unk>']
            self.length_dict = {}
            for i in self.nl_tokens:
                if len(i) not in self.length_dict:
                   self.length_dict.update({len(i): set()})
                self.length_dict[len(i)].add(i)
        self.length_list = sorted(list(self.length_dict.keys()), reverse=True)

    def tokenize_string(self,string_):
        """
        tokenize a string by standardizing whitespace around tokens (which are specified by list).
        this should be called before strings are fed into seq2seq model.
        go from the longest token to shortest. If we reached length of 1 and something is not yet tokenized,
        tokenize each char. In other words, all single-character tokens are automatically tokenized
        unless it's a part of a multi-character token.
        for Chinese tokens, if it's not one of the special ones listed below, we tokenize them all b/c their
        meaning doesn't matter.
        e.g. for input: 176=\frac{74}{UE}, the corresponding ind_marker (1 is add space, 2 is visited, 0 is unvisited)
        ind_marker = [0,0,0,0,1,2,2,2,2,0,0,0,0,0,0]
                      176=\frac{74}{UE}
        Returns:
            list of tokenized strings
        """

        token_list = []
        string_ = string_.replace(" ", "")
        i = 0
        while i < len(string_):
            found = False
            for token_length in self.length_list:
                if token_length > len(string_) - i:
                    continue
                substring = string_[i: i + token_length]
                if substring in self.length_dict[token_length]:
                    token_list.append(substring)
                    i += token_length
                    found = True
                    break
            if not found:
                token_list.append(string_[i: i + 1])
                i += 1
        return token_list

    def tokenize_string_ind(self, string_):
        token_list = []
        string_ = string_.replace(" ", "")
        i = 0
        while i < len(string_):
            found = False
            for token_length in self.length_list:
                if token_length > len(string_) - i:
                    continue
                substring = string_[i: i + token_length]
                if substring in self.length_dict[token_length]:
                    token_list.append(self.token2index[substring])
                    i += token_length
                    found = True
                    break
            if not found:
                token_list.append(self.unk_ind)
                i += 1
        return token_list

    def strip_token(self,string):
        tokenized_string = self.tokenize_string(string)
        for i, token in enumerate(tokenized_string):
            if token[0] == '\\':
                if len(token) > 1:
                    tokenized_string[i] = tokenized_string[i][1]
                else:
                    tokenized_string[i] = ""
        return ''.join(tokenized_string)

    def tokenlst2indlst(self, token_list):
        ind_list = []
        for token in token_list:
            if token in self.token2index.keys():
                ind_list.append(self.token2index[token])
            else:
                ind_list.append(self.unk_ind)
        return ind_list
    def indlst2tokenlst(self, ind_list):
        token_lst = []
        for ind in ind_list:
            token_lst.append(self.nl_tokens[ind])
        return token_lst
def tokenize_question_body(meta, tokenizer):
    for page_id, questions in meta.items():
        for question in questions:
            question_body = question['rawQuestionBody']
            new_question_body = []
            for question_body_line in question_body:
                new_question_body.append(tokenizer.tokenize_string(question_body_line))
            question['rawQuestionBody'] = new_question_body
    return meta


# from time import time
# lexicon_path = PipelineConfig.TYPE_LEXICON_PATH
# tk = Tokenizer(lexicon_path)
# t = time()
# print(tk.strip_token('efoijewioA.\\angleABP=\\angleCBP内部任取\\qqq簓疈'))
#
# print(time()-t)
# print()