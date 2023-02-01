# imports
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from Utilities import *
import seaborn as sns


def plot_box_plot(df):
    """ Plot box plots of the columns views, likes, dislikes, and comment_count"""
    fig, axes = plt.subplots(nrows=2, ncols=2)  # create 2x2 array of subplots
    # plot the data
    df.boxplot(column=['views'], ax=axes[0, 0], whis=[0, 100], showfliers=False)
    df.boxplot(column=['likes'], ax=axes[0, 1], whis=[0, 100], showfliers=False)
    df.boxplot(column=['dislikes'], ax=axes[1, 0], whis=[0, 100], showfliers=False)
    df.boxplot(column=['comment_count'], ax=axes[1, 1], whis=[0, 100], showfliers=False)
    fig.suptitle(df.name, fontsize=18)


def plot_word_cloud(df):
    """ plot word cloud of popular words in videos' titles """
    plt.rcParams['figure.figsize'] = (9, 9)
    plt.figure()
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=1000, height=1000,
                          max_words=121).generate(" ".join(df['title']))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Most Popular Words in Title (' + df.name + ')', fontsize=30)


def plot_pie(df):
    """ plot pie graph of number of comments disabled videos vs comments enabled """
    labels = 'Comments enabled', 'Comments disabled'
    sr = pd.Series(df['comments_disabled'])
    x = sr.value_counts()
    fig, ax = plt.subplots()
    ax.pie(x, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')
    fig.suptitle(df.name, fontsize=18)


def plot_likes_dislikes_views(df):
    """ scatter plot the likes and dislikes of videos vs their views"""
    ax = df.plot(x="views", y=["likes"], color='g', logx=True, kind="scatter", legend=True)
    df.plot(x='views', y=['dislikes'], color='r', logx=True, kind='scatter', legend=True, ax=ax, title=df.name)
    ax.legend(["likes", "dislikes"])
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)


def plot_videos_per_date(df):
    plt.figure()
    publish_dates = df['publish_time']
    publish_dates = publish_dates.apply(publish_str_to_date).tolist()
    publish_dates_no_dupes = list(dict.fromkeys(publish_dates))
    number_videos_per_date = [publish_dates.count(date) for date in publish_dates_no_dupes]

    lists = sorted(zip(*[publish_dates_no_dupes, number_videos_per_date]))
    new_x, new_y = list(zip(*lists))

    plt.plot(new_x, new_y)
    plt.title(df.name, fontsize=18)


def plot_videos_per_month(df):
    plt.figure()
    publish_dates = df['publish_time']
    publish_dates = publish_dates.apply(publish_str_to_date).tolist()
    months = [date.month for date in publish_dates]
    for month in range(1, 13):
        if month not in months:
            months.append(month)
    months_no_dupes = list(dict.fromkeys(months))
    number_videos_per_date = [months.count(date) for date in months_no_dupes]

    lists = sorted(zip(*[months_no_dupes, number_videos_per_date]))
    new_x, new_y = list(zip(*lists))

    plt.scatter(new_x, new_y)
    plt.plot(new_x, new_y)
    plt.xlabel("Month")
    plt.ylabel("Num videos")
    plt.xticks(range(1, 13))
    plt.title(df.name, fontsize=18)


def videos_per_hour(df):
    """ plot distribution of publishing times of trending videos"""
    fig, ax = plt.subplots(figsize=(10, 7))
    publish_hours = df['publish_time']
    publish_hours = publish_hours.apply(publish_str_to_hour)
    hours = ["%s:00" % (h) for h in range(24)]  # format x ticks labels
    x = np.arange(len(hours))
    ax.hist(publish_hours, bins=np.arange(25), rwidth=0.75, edgecolor='black', align='left')
    ax.set_xticks(x)  # set number of x ticks
    ax.set_xticklabels(hours)  # set labels
    plt.xticks(rotation=90)  # rotate it 90 degrees
    plt.title(df.name, fontsize=18)
    plt.xlabel("Hour")
    plt.ylabel("Num videos")


def plot_trending_time(df):
    """ Scatter plot the number of days it takes for a vid to hit trending by number of tags,
        plot an average line to help visualize results"""
    publish_dates = df['publish_time']
    trending_dates = df['trending_date']
    tags = df['tags']

    publish_dates = publish_dates.apply(publish_str_to_date)
    trending_dates = trending_dates.apply(trending_str_to_date)
    tags = tags.apply(count_tags)
    days_until_trending = ((trending_dates - publish_dates) / (np.timedelta64(1, 'D'))).astype('int64')

    plt.figure()
    c = Counter(zip(tags, days_until_trending))
    s = [7 * np.sqrt(c[(x, y)]) for x, y in zip(tags, days_until_trending)]
    plt.scatter(tags, days_until_trending, s=s, alpha=0.3)
    plt.autoscale()
    plt.xlabel("tags")
    plt.ylabel("days until trending")

    # plot avg line
    x, y = average_dots(tags.tolist(), days_until_trending.tolist(), s)

    plt.plot(x, y, 'r-')
    plt.title(df.name, fontsize=18)


def plot_distribution(df):
    # log all data
    df['likes_log'] = np.log(df['likes'] + 1)
    df['views_log'] = np.log(df['views'] + 1)
    df['dislikes_log'] = np.log(df['dislikes'] + 1)
    df['comment_log'] = np.log(df['comment_count'] + 1)

    f = plt.figure(figsize=(12, 6))

    plt.subplot(221)
    g1 = sns.distplot(df['views_log'])
    g1.set_title("VIEWS LOG DISTRIBUTION", fontsize=16)

    plt.subplot(224)
    g2 = sns.distplot(df['likes_log'], color='green')
    g2.set_title('LIKES LOG DISTRIBUTION', fontsize=16)

    plt.subplot(223)
    g3 = sns.distplot(df['dislikes_log'], color='r')
    g3.set_title("DISLIKES LOG DISTRIBUTION", fontsize=16)

    plt.subplot(222)
    g4 = sns.distplot(df['comment_log'])
    g4.set_title("COMMENTS LOG DISTRIBUTION", fontsize=16)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, top=0.9)
    f.suptitle(df.name, fontsize=18)
