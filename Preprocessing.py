from collections import Counter

class NgramTextNomalizer():
    def __init__(self, lexicon_file):
        self.unk = '<unk>'
        self.sent_start = '<s>'
        self.sent_end = '</s>'
        self.lexicon = self._get_fixed_lexicon(lexicon_file)

    def _get_fixed_lexicon(self, lexicon_file):
        sentences = self.sentence_separation(lexicon_file, 1, replacement=False, add_symbol=True)
        tockens = self.get_ngram_list(sentences, 1)[0]
        lexicon = [x[0] for x in Counter(tockens).keys()]
        lexicon = [self.sent_start] + lexicon
        return lexicon

    def sentence_separation(self, file_path, model_order, replacement, add_symbol):
        sentences = []
        with open(file_path) as f:
            if add_symbol:
                for line in f.readlines():
                    line = line.strip('\n')
                    line = line.lower()
                    line = line.split(' ')
                    BOS = (model_order - 1) * [self.sent_start]
                    EOS = [self.sent_end]
                    line = BOS + line + EOS
                    sentences.append(line)
            else:
                for line in f.readlines():
                    line = line.strip('\n')
                    line = line.lower()
                    line = line.split(' ')
                    sentences.append(line)
            if replacement:
                return self._replace_unknown(sentences)
            else:
                return sentences

    def get_ngram_list(self, sentences, model_order):
        '''
        To get every ngram as a tuple, notice the ngram may cross the sentence boundary
        :param sentences: preprocessed text
        :param n: model order
        :return: ngram_list: list->tuple->str and total number of ngrams
        '''
        ngram_list = []
        for sentence in sentences:
            for i in range(len(sentence) - model_order + 1):
                ngram_list.append(tuple(sentence[i:i + model_order]))
        return ngram_list, len(ngram_list)

    def _replace_unknown(self, sentences):
        replaced_sentences = []
        for sentence in sentences:
            replaced_sentence = []
            for tocken in sentence:
                if tocken not in self.lexicon:
                    replaced_sentence.append(self.unk)
                else:
                    replaced_sentence.append(tocken)
            replaced_sentences.append(replaced_sentence)
        return replaced_sentences

if __name__ == '__main__':
    ntn = NgramTextNomalizer('Europal-v9')
    sentences = ntn.sentence_separation('wsj.text.test', 5, replacement=True, add_symbol=True)
