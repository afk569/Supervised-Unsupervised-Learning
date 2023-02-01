from datetime import datetime


def average_dots(x, y, weights):
    """ get lists x,y and weights, return x without duplicates and weighted mean of y with the weights"""
    sorting_indices = sorted(range(len(x)), key=lambda k: x[k])  # get sorting indices
    x = sorted(x)  # sort normally
    y = [y[i] for i in sorting_indices]  # sort according to x

    weights = [weights[i] for i in sorting_indices]  # sort according to x
    x_no_dupes = list(dict.fromkeys(x))
    y_avg = []

    # calculate the weighted mean
    for i in range(len(x_no_dupes)):
        indices = [k for k, b in enumerate(x) if b == x_no_dupes[i]]  # get indices of the duplicates
        y_avg.append(sum([y[j] * (weights[j] / 1) for j in indices]) / sum([(weights[j] / 1) for j in indices]))

    return x_no_dupes, y_avg


def trending_str_to_date(str):
    """ convert the string format in trending column to date object """
    return datetime.strptime(str, '%y.%d.%m')


def publish_str_to_hour(str):
    """ convert the string format in publish column to hour (int) """
    date = datetime.strptime(str, "%Y-%m-%dT%H:%M:%S.000z")
    return date.hour


def publish_str_to_date(str):
    """ convert the string format in publish column to date object """
    str = str[:10]
    return datetime.strptime(str, '%Y-%m-%d')


def count_tags(tags_string):
    """ count tags in the string format in the column """
    if '|' not in tags_string:
        return 0
    return tags_string.count('|') + 1


def get_name_from_source(source):
    """ get string in format 'XX'videos.csv and get the XX """
    source_dict = {"US": "USA", "GB": "Great Britain", "DE": "Germany", "CA": "Canada", "FR": "France", "RU": "Russia",
                   "MX": "Mexico", "KR": "South Korea", "JP": "Japan", "IN": "India"}

    return source_dict.get(source.replace("videos.csv", "").replace("Data/", ""))
