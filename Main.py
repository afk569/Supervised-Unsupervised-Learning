# imports
from easygui import *
from Algorithms import *
from Statistics import *
from Tests import *


def main():
    # set pandas float format
    pd.options.display.float_format = '{:,.5f}'.format

    # dicts
    countries_dict = {"USA": "US", "Great Britain": "GB", "Germany": "DE", "Canada": "CA", "France": "FR",
                      "Russia": "RU", "Mexico": "MX", "South Korea": "KR", "Japan": "JP", "India": "IN"}
    algo_dict = {"K-Means": plot_kMeans,
                 "Fuzzy-C-Means": plot_fcm,
                 "GMM": plot_gmm,
                 "DBScan": plot_dbscan,
                 "PCA (views, likes, dislikes, comments count)": plot_PCA,
                 "ICA (views, likes, dislikes, comments count)": plot_ICA,
                 "KPCA (views, likes, dislikes, comments count)": plot_KPCA}
    scores_dict = {"K-Means": calc_kMeans,
                   "Fuzzy-C-Means": calc_fcm,
                   "GMM": calc_gmm,
                   "DBScan": calc_dbscan}
    stats_dict = {"Box-plot": plot_box_plot,
                  "Word Cloud": plot_word_cloud,
                  "Pie": plot_pie,
                  "Likes & Dislikes vs Views": plot_likes_dislikes_views,
                  "Videos per date": plot_videos_per_date,
                  "Videos per month": plot_videos_per_month,
                  "Videos per hour": videos_per_hour,
                  "Trending time by tags": plot_trending_time,
                  "Data distribution": plot_distribution}

    picked_countries = multchoicebox("Pick countries for data", choices=countries_dict.keys())
    if picked_countries is not None:
        countries_codes = [countries_dict.get(country) for country in picked_countries]
        while countries_codes is not None and len(countries_codes) > 0:
            plot_scores = False
            csv_sources = ["Data/" + x + "videos.csv" for x in countries_codes]
            dataframes = read_csv(csv_sources)

            category_choices = ["Algorithms", "Statistics", "Tests", "Plot scores"]
            category = choicebox("Choose category:", choices=category_choices)
            while not (category is None):
                if category == "Plot scores":
                    domain_values = []
                    domain_values = multenterbox("Range of number of clusters?",
                                                 fields=["Min num of clusters", "Max num of clusters"],
                                                 values=domain_values)
                    domain = range(max(0, int(domain_values[0])), min(5, int(domain_values[1])) + 1)
                    plot_scores = True
                    msgbox(
                        "Scores plot will be shown after plotting an algorithm (and closing the window) \n"
                        + " * notice that it might time some time if the range is big")
                if category == "Algorithms":
                    # choose algo
                    algo = choicebox("Which algorithm?", choices=algo_dict.keys())

                    if algo in algo_dict:
                        run_function(algo_dict, algo, dataframes)
                    # it's likes that because of a bug in the GUI module
                    if (algo in algo_dict) & plot_scores:
                        run_plot_scores(scores_dict, algo, dataframes, domain)
                elif category == "Statistics":
                    # choose stats
                    stat = choicebox("Which statistic?", choices=stats_dict.keys())

                    if stat in stats_dict:
                        run_function(stats_dict, stat, dataframes)
                elif category == "Tests":
                    # run T test
                    t_test_countries = multchoicebox("Choose 2 countries", choices=picked_countries)
                    if t_test_countries is not None:
                        column = choicebox("Choose column", choices=list(dataframes[0]))
                        selected_data = [dataframe[column] for dataframe in dataframes if
                                         dataframe.name in t_test_countries]
                        if len(selected_data) > 1:
                            alpha = float(enterbox("What's your alpha?", default='0.05'))
                            t_test(selected_data[0].mean(), selected_data[1].mean(),
                                   selected_data[0].std(), selected_data[1].std(),
                                   selected_data[0].size, selected_data[1].size,
                                   alpha)

                category = choicebox("Choose category:", choices=category_choices)

            # reset and rerun
            countries_codes = []
            picked_countries = multchoicebox("Pick countries for data", choices=countries_dict.keys())
            if picked_countries is not None:
                countries_codes = [countries_dict.get(country) for country in picked_countries if not (country is None)]


def run_function(choice_dict, choice, dataframes):
    for df in dataframes:
        (choice_dict.get(choice))(df)
    plt.show()


def run_plot_scores(choice_dict, choice, dataframes, domain):
    for df in dataframes:
        plot_cluster_scores(choice_dict.get(choice), df, domain)
    plt.show()


def read_csv(sources):
    """ Read csv files into DataFrames list """
    dataframes = []
    for i in range(len(sources)):
        try:
            df = pd.read_csv(sources[i], sep=',\s+', delimiter=',', encoding="utf-8", skipinitialspace=True)
            df.name = get_name_from_source(sources[i])
            dataframes.append(df)
        except:
            try:
                df = pd.read_csv(sources[i], sep=',\s+', delimiter=',', encoding="ISO-8859-1", skipinitialspace=True)
                df.name = get_name_from_source(sources[i])
                dataframes.append(df)
            except:
                print("Can't understand encoding")

    return dataframes


if __name__ == '__main__':
    main()
