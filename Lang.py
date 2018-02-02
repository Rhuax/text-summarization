class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0,"EOS":1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def getIdFromWord(self,word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return -1
    def getWordFromId(self,id):
        if id in self.index2word:
            return self.index2word[id]
        else:
            return None
