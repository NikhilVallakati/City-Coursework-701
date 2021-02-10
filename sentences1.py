from word_processing import Word_Processing1

class Sentences1:

    @staticmethod
    def filter(cL):
        sentences1 = []
        s = ' '
        for c in cL:
            words = c.split()
            words = map(lambda x: Word_Processing1.clean_word(x), words) # Remove "stop" words that do not influence sentiment
            words = list(filter(lambda x:True if len(x) > 0 else False, words))
            wordline = s.join(words)
            sentences1.append(wordline)
            
        return sentences1

    @staticmethod
    def filter1(cL):
        sentences1 = []
        s = ' '
        for c in cL:
            words = c[0].split()
            words = map(lambda x: Word_Processing1.clean_word(x), words) # Remove "stop" words that do not influence sentiment
            words = list(filter(lambda x:True if len(x) > 0 else False, words))
            wordline = s.join(words)
            sentences1.append(wordline)
            
        return sentences1