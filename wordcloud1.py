import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

class Wordcloudz:

    @staticmethod
    def cloud_plot(wordcloud1):
        plt.figure(1, figsize=(20,15))
        plt.imshow(wordcloud1)
        plt.axis('off')
        plt.show()

    @staticmethod
    def show(dataset_, column_):
        wordcloud2 = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=250,
                          max_font_size=40, 
                         ).generate(str(dataset_[column_].dropna()))
        Wordcloudz.cloud_plot(wordcloud2)


    