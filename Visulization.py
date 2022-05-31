import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import nltk

emotions = ['fear', 'anger', 'anticipation', 'trust', 'surprise','sadness', 'disgust', 'joy']
class Visulization:
    @staticmethod
    def createSentimentPieChart(tweets,title):
        props = tweets['sentiment'].value_counts(normalize=True)
        #print(props)
        plt.figure()
        plt.pie(props,labels=props.keys(),autopct='%.0f%%',)
        plt.title(title)
        return props

    @staticmethod
    def createEmotionPieChart(tweets,title):
        res = {}
        for emotion in emotions:
            res[emotion] = len(tweets.query(emotion+'==1'));
        print(res)
        plt.figure()
        plt.pie(res.values(),labels=res.keys(),autopct='%.0f%%',)
        plt.title(title)

    @staticmethod
    def createWordCloudForEmotion(tweets,emotion,column,title):
        filtered_tweets = tweets.query(emotion+'==1')

        #tokens = sum(res_map, [])
        # Create a WordCloud object
        wordcloud = WordCloud(mode="RGBA", background_color=None , max_words=1000, height = 400, width = 900, contour_width=3, contour_color='steelblue')
        long_string = Visulization.create_joint_string(filtered_tweets,column)
        # Generate a word cloud
        wordcloud.generate(long_string)
        # Visualize the word cloud
        wordcloud.to_image()
        plt.figure(figsize=(20,15))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title(title, fontsize=18, color = 'Orange')
        plt.show()
    @staticmethod
    def create_joint_string(tweets,column):
        res_map = list(map(Visulization.Convert,tweets[column].values))
        long_string = ' '.join(list(map(' '.join,res_map)))
        return long_string
    @staticmethod
    def plot_frequency_chart(info):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.set_context("notebook", font_scale=1)
        ax = sns.barplot(x=info['x'], y=info['y'], data=info['data'], palette=(info['pal']))
        ax.set_title(label=info['title'], fontweight='bold', size=18)
        plt.ylabel(info['ylab'], fontsize=16)
        plt.xlabel(info['xlab'], fontsize=16)
        plt.xticks(rotation=info['angle'],fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        #plt.savefig('images/' + info['fname'])

        return
    @staticmethod
    def creatNGramsChart(tweets,emotion,title = 'default title',n=3,topCnt = 10):
        filtered_tweets = tweets.query(emotion+'==1')
        assert (n == 2 or n==3)
        long_string = Visulization.create_joint_string(filtered_tweets,'tokens')
        # Get tokens
        tokens = long_string.split(' ')
        if n == 3:
            trigrams = nltk.trigrams(tokens)
        elif n == 2:
            trigrams = nltk.trigrams(tokens)
        df_trigrams = Visulization.get_top_n_grams(trigrams,n,topCnt)
        info = {'data': df_trigrams, 'x': 'Grams', 'y': 'Count',
                'xlab': 'Trigrams', 'ylab': 'Count', 'pal':'viridis',
                'title': title,
                'angle': 40}
        Visulization.plot_frequency_chart(info)

    @staticmethod
    def get_top_n_grams(trigrams, N=3, top_grams = 10):
        grams_str = []
        data = []
        gram_counter = Counter(trigrams)

        for grams in gram_counter.most_common(top_grams):
            gram = ''
            grams_str = grams[0]
            grams_str_count = []
            for n in range(0,N):
                gram = gram + grams_str[n] + ' '
            grams_str_count.append(gram)
            grams_str_count.append(grams[1])
            data.append(grams_str_count)
        df = pd.DataFrame(data, columns = ['Grams', 'Count'])

        return df
    @staticmethod
    def Convert(string):
        # li = list(string.strip('][').split(", "))
        # res = [x.strip('\'\'') for x in li]
        # if string.isnull()    :
        #     return []
        res = list(string.split(" "))
        return res