import math
from collections import Counter
from Preprocessing import NgramTextNomalizer

class NgramModel:
    def __init__(self,n,ngram_list):
        self.unk = ['<unk>']
        self.bos = ['<s>']
        self.eos = ['</s>']
        self.model_order = n
        self.lm, self.unk_prob = self.train(ngram_list)

    def train(self,ngram_list):
        ngram_count = Counter(ngram_list) #dict{tuple->str: int}
        kgram_count_list = self._count_adj_kgram(ngram_count)
        probs = self._cal_prob(kgram_count_list)
        unk_prob = probs[4][('<unk>',)]
        return probs, unk_prob

    def _count_adj_kgram(self,ngram_count):
        '''
        To get the adjusted count(from the second highest order)
        the count for the highest order remains original
        :param ngram_count: dict{tuple->str: int}
        :return: list->dict{tuple->str: int} with decreasing order
        '''
        kgram_count_list = [ngram_count]
        for i in range(1, self.model_order):
            last = kgram_count_list[-1]
            new = {}
            for kgram in last.keys():
                new[tuple(kgram[1:])] = new.get(tuple(kgram[1:]), 0) + 1
            kgram_count_list.append(new)
        return kgram_count_list

    def _calculate_discount_list(self, kgram_count_count):
        '''
        For every kgram there are 3 discounting factors according to it specific count
        :param kgram_count_count: dict{int: int}
        :return:
        '''
        Y_k = kgram_count_count[1] / (kgram_count_count[1] + 2 * kgram_count_count[2])
        discount_list_k = [0]
        for i in range(1, 4):
            if kgram_count_count[i] == 0:
                discount_k = 0
            else:
                discount_k = i - (i+1) * Y_k * (kgram_count_count[i+1] / kgram_count_count[i])
            discount_list_k.append(discount_k)
        return discount_list_k

    def _get_discount(self, discount_list_k, count):
        if count > 3:
            return discount_list_k[3]
        else:
            return discount_list_k[count]

    def _cal_bow_adj_prob(self, order):
        '''
        back off weight for specific order
        :param order:
        :return: k_th back off weight of each prefix dict{'str': float} and the adjusted prob dict{'str': float}
        '''
        kgram_count_count = Counter(value for value in order.values() if value <= 4)
        discounts = self._calculate_discount_list(kgram_count_count)
        bow = {}
        order_adj_prob = order
        prefix_sum = {}
        for kgram in order.keys():
            prefix = kgram[: -1]
            count = order[kgram]
            prefix_sum[prefix] = prefix_sum.get(prefix, 0) + count
            D = self._get_discount(discounts, count)
            bow[prefix] = bow.get(prefix, 0) + D
            order_adj_prob[kgram] = order_adj_prob.get(kgram, 0) - D
        for kgram in order_adj_prob.keys():
            prefix = kgram[: -1]
            order_adj_prob[kgram] = order_adj_prob[kgram]/prefix_sum[prefix]
        for prefix in bow.keys():
            bow[prefix] = bow[prefix]/prefix_sum[prefix]
        return bow, order_adj_prob

    def _cal_unigram_prob(self, unigram_count):
        unigram_prob = {}
        prefix_sum = sum(count for count in unigram_count.values())
        for key in unigram_count.keys():
            unigram_prob[key] = unigram_count[key]/prefix_sum
        #unigram_prob = dict((k,math.log(v/prefix_sum)) for k,v in unigram_count.items())
        return unigram_prob

    def _cal_prob(self, orders):
        bow_list = []
        order_adj_prob_list = []
        probs = []
        for order in orders[:-1]:
            bow, order_adj_prob = self._cal_bow_adj_prob(order)
            bow_list.append(bow)
            order_adj_prob_list.append(order_adj_prob)
        order_adj_prob_list.append(self._cal_unigram_prob(orders[-1]))
        bow_list.reverse()
        order_adj_prob_list.reverse()
        last_order = order_adj_prob_list[0]
        for order, bow in zip(order_adj_prob_list[1:], bow_list):
            for kgram in order.keys():
                prefix = kgram[:-1]
                suffix = kgram[1:]
                order[kgram] = order.get(kgram, 0) + (last_order[suffix] * bow[prefix])
            last_order = order
            probs.append(order)
        probs.reverse()
        probs.append(order_adj_prob_list[0])
        return probs

    def logprob(self, ngram):
        for i, order in enumerate(self.lm):
            if ngram[i:] in order:
                return math.log(order[ngram[i:]])
        return self.unk_prob

    def sentence_prob(self, sentence):
        '''
        :param sentence: list->str
        :return:
        '''
        adj_sentence = tuple(self.bos *(self.model_order-1) + sentence + self.eos)
        sent_logprob = 0
        for i in range(len(adj_sentence)-self.model_order+1):
            ngram = adj_sentence[i:i+self.model_order]
            sent_logprob += self.logprob(ngram)
        return sent_logprob

    def perplexity(self, sentences, size):
        pp = 0
        for sentence in sentences:
            pp += self.sentence_prob(sentence)
        pp = -(1/size) * pp
        return math.exp(pp)

if __name__ == '__main__':
    ntn = NgramTextNomalizer('Europal-v9')
    train = 'wsj.text.train'
    test = 'wsj.text.test'
    training_text = ntn.sentence_separation(train, 5, replacement=True, add_symbol=True)
    test_text = ntn.sentence_separation(test, 5, replacement=True, add_symbol=False)
    training_ngram_list, training_size = ntn.get_ngram_list(training_text, 5)
    test_ngram_list, test_size = ntn.get_ngram_list(test_text, 5)
    lm = NgramModel(5, training_ngram_list)
    #print(lm.sentence_prob(['today','is','a','good','day','.']))
    print(lm.perplexity(test_text, test_size))
