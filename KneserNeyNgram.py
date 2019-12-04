import math
from collections import defaultdict, Counter

def preprocessing(file,model_order):
    sentence = []
    with open(file) as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.lower()
            line = line.split(' ')
            BOS = (model_order-1) * ['bos']
            EOS = ['eos']
            line = BOS + line + EOS
            sentence.append(line)
    return sentence

def get_ngram_list(text,n):
    ngram_list = []
    for sentence in text:
        for i in range(len(sentence)-n+1):
            ngram_list.append(tuple(sentence[i:i+n]))
    return ngram_list


class NgramModel:
    def __init__(self,n,ngram_list):
        self.bos = ['bos']
        self.eos = ['eos']
        self.model_order = n
        self.lm = self.train(ngram_list)

    def train(self,ngram_list):
        ngram_count = Counter(ngram_list)
        kgram_count_list = self._count_adj_kgram(ngram_count)  #Counter(ngram_list)返回ngram_count
        probs = self._cal_prob(kgram_count_list)
        return probs

    def _count_adj_kgram(self,ngram_count):
        '''
        kgram_count: a list for back-off gram, every element is a gram count dict of each order
        '''
        kgram_count_list = [ngram_count]
        for i in range(1,self.model_order):
            last_order = kgram_count_list[-1]
            new_order = defaultdict(int)
            for kgram in last_order.keys():
                new_order[tuple(kgram[1:])] = new_order.get(tuple(kgram[1:]),0) + 1
            kgram_count_list.append(new_order)
        return kgram_count_list

    def _calculate_discount_list(self,kgram_count_count):
        '''
        下面是计算discount
        kgram_count_count is a dict,
        对于一个kgram来说,他的D只有三种可能,D1,D2,D3+
        '''
        Y_k = kgram_count_count[1] / (kgram_count_count[1] + 2 * kgram_count_count[2])
        discount_list_k = [0]
        for i in range(1,4):
            if kgram_count_count[i] == 0:
                discount_k = 0
            else:
                discount_k = i - (i+1) * Y_k * (kgram_count_count[i+1]/kgram_count_count[i])
            discount_list_k.append(discount_k)
        return discount_list_k

    def _get_discount(self,discount_list_k,count):
        if count > 3:
            return discount_list_k[3]
        else:
            return discount_list_k[count]

    def _cal_backoff_weight(self, order):
        '''
        back off weight for specific order
        :param kgram_count:
        :return: back off weight of each prefix
        '''
        kgram_count_count = Counter(value for value in order.values() if value <= 4)
        discounts = self._calculate_discount_list(kgram_count_count)
        bow = defaultdict(int)
        order_adj_prob = order
        prefix_sum = defaultdict(int)
        for kgram in order.keys():
            prefix = kgram[:-1]
            count = order[kgram]
            prefix_sum[prefix] = prefix_sum.get(prefix,0) + count
            discount = self._get_discount(discounts,count)
            bow[prefix] = bow.get(prefix,0) + discount
            order_adj_prob[kgram] = order_adj_prob.get(kgram,0) - discount
        for kgram in order_adj_prob.keys():
            prefix = kgram[:-1]
            order_adj_prob[kgram] = math.log(order_adj_prob[kgram]/prefix_sum[prefix])
        for prefix in bow.keys():
            bow[prefix] = math.log(bow[prefix]/prefix_sum[prefix])
        return bow, order_adj_prob

    def _cal_unigram_prob(self,unigram_count):
        sum_count = sum(count for count in unigram_count.values())
        unigram = dict((k,math.log(v/sum_count)) for k,v in unigram_count.items())
        return unigram

    def _cal_prob(self,orders):
        bow_list = []
        order_adj_prob_list = []
        probs = []
        for order in orders[:-1]:
            bow, order_adj_prob = self._cal_backoff_weight(order)
            bow_list.append(bow)
            order_adj_prob_list.append(order_adj_prob)
        #bow_list.append(defaultdict(int))
        order_adj_prob_list.append(self._cal_unigram_prob(orders[-1]))
        for last_order, order, bow in zip(reversed(order_adj_prob_list), reversed(order_adj_prob_list[:-1]), reversed(bow_list[:-1])):
            for kgram in order.keys():
                prefix = kgram[:-1]
                suffix = kgram[1:]
                order[kgram] = order.get(kgram,0) + (last_order[suffix] + bow[prefix])
            probs.append(order)
        probs.insert(0,order_adj_prob_list[-1])
        return reversed(probs)

    def logprob(self,ngram):
        for i, order in enumerate(self.lm):
            if ngram[i:] in order:
                return order[ngram[i:]]
        return None

    def sentence_prob(self,sentence):
        adj_sentence = tuple(self.bos *(self.model_order-1) + sentence + self.eos)
        sent_logprob = 0
        for i in range(len(adj_sentence)-self.model_order+1):
            ngram = adj_sentence[i:i+self.model_order]
            sent_logprob += self.logprob(ngram)
        return math.pow(10,sent_logprob)

if __name__ == '__main__':
    file = 'Europal-v9'
    training_text = preprocessing(file,5)
    ngram = get_ngram_list(training_text,5)
    lm = NgramModel(5,ngram)
    print(lm.sentence_prob(['i','have','a','dream','.']))
