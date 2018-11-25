import locale
from pprint import pprint
from typing import Optional, List, Dict, Any, Tuple
from typing import List, Dict, Any
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions

import pytz
import twitter
import datetime
import json

from dataset.constants import RETWEETED_STATUS_KEY, USER_KEY, SCREEN_NAME_KEY, USER_MENTIONS_KEY, \
    IN_REPLY_TO_SCREEN_NAME_KEY, RETWEETS_KEY, MENTIONS_KEY, COMMENTS_KEY, DESCRIPTION_SIMILARITY, FOLLOWING_SIMILARITY, \
    DATE_OF_CREATION_SIMILARITY, HASHTAGS_SIMILARITY, CATEGORIES_SIMILARITY, INTERACTION_VECTOR_KEY, \
    SIMILARITY_VECTOR_KEY, NUMBER_OF_COMMENTS_KEY
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

consumer_id_key = 'NC19WDNaMoEaaV9s8nadVUvBI'
consumer_secret_key = 'rtoYVT9AykdYQWWv5Nh7ZdDode72DRSLX8XswRrqprxhC2TnI3'
access_token_key = '1058944938887409664-402KNxXTgNNhMoAK3pmAUziB3Fhxzj'
access_token_secret = 'vVxGZQdN9ubMK1EnxZGx62ZsxVqwFQiwDZmXmEMfPYSKj'
api = None


class TwitterApi():
    def get_api_instance(self):
        global api
        if api:
            return api
        api = twitter.Api(consumer_key=consumer_id_key,
                          consumer_secret=consumer_secret_key,
                          access_token_key=access_token_key,
                          access_token_secret=access_token_secret)
        api.VerifyCredentials()
        return api

    def find_posts_twitter(self, api, screen_name: str, pool_amount: int, since: datetime,
                           until: datetime = datetime.datetime.now(datetime.timezone.utc)) -> List[Dict[str, Any]]:
        """
        Finds user's posts and sorts them b amount of likes
        :param api: instance of twitter api
        :param screen_name: twitter screen name of a user
        :param pool_amount: amount of posts to retrieve (if 0 we retrieve posts by time period [since; until])
        :param since: datetime of oldest post (is used only if pool amount is not mentioned)
        :param until: datetime of latest post (is used only if pool amount is not mentioned)
        :return: list of post dictionaries
        """

        if not api:
            return []
        if not since:
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        posts = []
        if pool_amount is None or pool_amount == 0:
            pool_amount = 20

        try:
            if since and since > datetime.datetime.now(datetime.timezone.utc):
                since = datetime.datetime.now(datetime.timezone.utc)
            if until and until > datetime.datetime.now(datetime.timezone.utc):
                until = datetime.datetime.now(datetime.timezone.utc)

            if since and until and since > until:
                return []

            last_id = ''
            count = 0
            search_finish = False  # flag that indicates that search is finished
            first_cycle = True
            while (count < pool_amount or pool_amount == -1) and not search_finish:
                count_to_request = 200
                time_line = api.GetUserTimeline(screen_name=screen_name, count=count_to_request, max_id=last_id)
                if not time_line:
                    break

                # popping first element from timeline as it was received in previous query
                if not first_cycle:
                    time_line.pop(0)

                first_cycle = False

                # adding posts from timeline to posts that satisfy required timeperiod
                for status in time_line:
                    post = status.AsDict()
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        last_id = post['id_str']
                        date = post['created_at']
                        date_parsed = self.get_tweet_date(date)
                        post['created_at'] = self.serialize_datetime(date_parsed)
                        posts.append(post)
                        count += 1

                    elif posts:
                        last_id = post['id_str']
                        search_finish = True
                        break

            sorted_posts = sorted(posts, key=lambda x: int(x.get('favorite_count', 0)), reverse=True)
            if pool_amount != -1:
                result = sorted_posts[:pool_amount]
            else:
                result = sorted_posts
        except Exception as e:
            print(f'Exception occured: {e}')
            return []

        return result

    def find_retweets_twitter(self, posts: List[Dict], screen_name: str) -> List[Dict[str, Any]]:
        """
        Finds retweets from given user's screen_namelist in given list of posts
        :param posts: list of posts in some user timeline
        :param screen_name: user that was an author of tweet that was retweeted
        :return: list of retweet dictionaries
        """
        retweets = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                if author_screen_name == screen_name:
                    retweets.append(post)
        return retweets

    def find_mentions_twitter(self, posts: List[Dict], screen_name: str) -> List[Dict[str, Any]]:
        """

        :param posts:
        :param screen_name:
        :return:
        """
    def get_posts_information(self, posts: List[Dict]) -> Tuple[List, List, List]:
        """
        Gets a list of all users that were mentioned, commented and retweeted in given posts
        :param posts:
        :return:
        """
        retweeted_users = []
        mentioned_users = []
        commented_users = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                retweeted_users.append(author_screen_name)
            if USER_MENTIONS_KEY in post:
                if IN_REPLY_TO_SCREEN_NAME_KEY in post:
                    reply_screen_name = post.get(IN_REPLY_TO_SCREEN_NAME_KEY, '')
                    commented_users.append(reply_screen_name)
                else:
                    users_mentioned = post.get(USER_MENTIONS_KEY, [])
                    for user in users_mentioned:
                        user_screen_name = user.get(SCREEN_NAME_KEY, '')
                        commented_users.append(user_screen_name)
        return retweeted_users, mentioned_users, commented_users


    def find_user_interactions_in_posts(self, posts: List[Dict], screen_name: str) -> Dict[str, Any]:
        """

        :param posts:
        :param screen_name:
        :return:
        """
        interactions = {}
        retweets = []
        mentions = []
        comments = []
        for post in posts:
            if RETWEETED_STATUS_KEY in post:
                retweet_info = post.get(RETWEETED_STATUS_KEY, {})
                retweet_author = retweet_info.get(USER_KEY, {})
                author_screen_name = retweet_author.get(SCREEN_NAME_KEY, '')
                if author_screen_name == screen_name:
                    retweets.append(post)
                    continue  # not to add retweets to mentions
            if USER_MENTIONS_KEY in post:
                if IN_REPLY_TO_SCREEN_NAME_KEY in post:
                    reply_screen_name = post.get(IN_REPLY_TO_SCREEN_NAME_KEY, '')
                    if reply_screen_name == screen_name:
                        comments.append(post)
                else:
                    users_mentioned = post.get(USER_MENTIONS_KEY, [])
                    for user in users_mentioned:
                        user_screen_name = user.get(SCREEN_NAME_KEY, '')
                        if user_screen_name == screen_name:
                            mentions.append(post)

        return {RETWEETS_KEY: retweets,
                MENTIONS_KEY: mentions,
                COMMENTS_KEY: comments}

    def get_tweet_date(self, date: str):
        try:
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
            return datetime.datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y')
        except Exception as e:
            print(f'error has occured: {e}')

    def serialize_datetime(self, date: datetime) -> str:
        """
        Make a string representing a datetime in UTC. The resulting string
        can be deserialized by deserialize_datetime().

        If the datetime given is naive or is not in UTC, then exception is raised.

        Milliseconds can be lost in serialization.

        :param date: UTC datetime to serialize
        :return: datetime string
        """
        if date.tzinfo not in [pytz.UTC, datetime.timezone.utc]:
            raise ValueError('Datetime should not be naive and should be in UTC (pytz.UTC or datetime.timezone.utc)')
        return date.strftime('%Y-%m-%dT%H:%M:%S%z')

    def a_is_follower_of_b(self, api, screen_name_a: str, screen_name_b: str):
        """
        Is screen_name_1 subscriber of screen_name_2?
        :return: True or False depending on
        """

        if not api:
            return -1
        try:
            followers = api.GetFriends(screen_name=screen_name_a)

            for user in followers:
                if user.screen_name == screen_name_b:
                    return True

            return False
        except Exception as e:
            print(f'Exception occured: {e}')
        return -1

    def get_followers_of(self, api, screen_name):
        if not api:
            return -1
        try:
            return api.GetFriends(screen_name=screen_name)

        except Exception as e:
            print(f'Exception occured: {e}')
        return -1

    # S
    def similarity_creation_date(self, api, screen_name_1, screen_name_2):
        """
        The absolute value of difference. max is 1, min is lim->0. Counts as 1/Absolute_Value[user_date_1 - user_date_2]
        :return: max 1 | min 0
        """

        if not api:
            return -1
        try:
            user_1 = api.GetUser(screen_name=screen_name_1)
            user_2 = api.GetUser(screen_name=screen_name_2)

            date_user_1 = self.get_tweet_date(date=user_1.created_at)
            date_user_2 = self.get_tweet_date(date=user_2.created_at)
            return 1 / (abs(date_user_2 - date_user_1).days)
        except Exception as e:
            print(f'Exception occured: {e}')

        return -1

    # S TODO check
    def common_subscriptions_a_to_b(self, api, screen_name_a: str, screen_name_b: str):
        """
        If user 1 and user 2 are subscribed to one account, the name of the common acc. will be added to result list
        """
        if not api:
            return -1
        try:
            friends_1 = api.GetFriends(screen_name=screen_name_a)
            friends_2 = api.GetFriends(screen_name=screen_name_b)

            common_list = []

            for friend_1 in friends_1:
                for friend_2 in friends_2:
                    if friend_2.screen_name == friend_1.screen_name:
                        common_list.append(friend_1.screen_name)

            return len(common_list) / len(friends_1)
        except Exception as e:
            print(f'Exception occured: {e}')

        return -1

    def get_favorites_users(self, api, screen_name, since: datetime,
                            until: datetime = datetime.datetime.now(datetime.timezone.utc))-> List[str]:
        """
        Returns list of people whos posts were liked
        :param api:
        :param screen_name:
        :param screen_name2:
        :param since:
        :param until:
        :return: integer number indicating how many times screen_name liked screen name2
                -1 on error
        """
        if not api:
            return -1

        try:
            since, until = self.parse_time_interval(since, until)
            if since and until and since > until:
                return -1

            search_done = False
            last_id = ''
            count = 0
            favorites_authors = []
            while not search_done:
                favorites = api.GetFavorites(screen_name=screen_name, count=200, since_id=last_id)
                for fav in favorites:
                    post = fav.AsDict()
                    last_id = post['id_str']
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        post_author = post['user']['screen_name']
                        favorites_authors.append(post_author)
                    elif since > self.get_tweet_date(post['created_at']):
                        search_done = True
                        break

            return favorites_authors
        except Exception as e:
            print(f'Exception occured: {e}')
            return -1

    def parse_time_interval(self, since, until: datetime):
        if not since:
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

        if since and since > datetime.datetime.now(datetime.timezone.utc):
            since = datetime.datetime.now(datetime.timezone.utc)
        if until and until > datetime.datetime.now(datetime.timezone.utc):
            until = datetime.datetime.now(datetime.timezone.utc)

        return since, until

    # S
    def get_hashtags_similarity(self, posts1, posts2):
        """
        Calculates similarity of posts of two users based on hashtags
        :param posts1: list of posts of 1st user
        :param posts2: list of posts of user to compare with
        :return: percentage(number > 0 and < 1) of common hashtags w.r.t the 1st user
        """
        tags1 = []
        for post in posts1:
            for tag in post['hashtags']:
                tags1.append(tag)
        tags2 = []
        for post in posts2:
            for tag in post['hashtags']:
                tags2.append(tag)

        if not len(tags1) or not len(tags2):
            return 0
        count = 0
        for tag1 in tags1:
            for tag2 in tags2:
                if tag1 == tag2:
                    count += 1
        return count / len(tags1)

    def _get_natural_language_understanding(self, version):
        n = NaturalLanguageUnderstandingV1(
            version=version,
            iam_apikey='SGjJgUAGXQEdbXiRe27u2V4hmeMIrEESo0vcXrfCunLL',
            url='https://gateway-wdc.watsonplatform.net/natural-language-understanding/api'
        )
        return n

    def _get_categories(self, posts):
        """
        :param posts: list of posts to analyze
        :return: list of unique categories
        """
        if not len(posts):
            return []
        natural_language_understanding = self._get_natural_language_understanding('2018-03-16')
        categories = []
        for post in posts:
            try:
                response = natural_language_understanding.analyze(
                    text=post['text'],
                    features=Features(categories=CategoriesOptions())).get_result()
                for category in response['categories']:
                    for c in category['label'].split('/'):
                        if c == '' or c in categories:
                            continue
                        categories.append(c)
            except Exception:
                continue
        return categories

    # S
    def get_categories_similarity(self, categories1, categories2):
        """
         Calculates users' similarity of posts based on categories from their posts
        :param posts1: list of posts of the 1st user
        :param posts2: list of posts of the user to compare with
        :return: percentage(number > 0 and < 1) of common categories w.r.t to the 1st user
        """

        if not len(categories1) or not len(categories2):
            return 0
        count = 0
        for category1 in categories1:
            for category2 in categories2:
                if category1 == category2:
                    count += 1
        return count / len(categories1)

    def get_all_hashtags(self, posts):
        hashtags = []
        for post in posts:
            hashtags.append(post['hashtags'])
        return hashtags



class DatasetCollection():
    def save_posts_of_user(self, twitterSearch: TwitterApi, screen_names, file):
        all_users = []
        for screen_name in screen_names:

            posts = twitterSearch.find_posts_twitter(api=twitterSearch.get_api_instance(), screen_name=screen_name,
                                                     pool_amount=-1,
                                                     since=datetime.datetime.now(
                                                         datetime.timezone.utc) - datetime.timedelta(31),
                                                     until=datetime.datetime.now(
                                                         datetime.timezone.utc))  # TODO WTF is this time

            # retweets = twitterSearch.find_retweets_twitter(posts=posts, screen_name=screen_name)
            retweeted_users, mentioned_users, commented_users = twitterSearch.get_posts_information(posts)
            favorite_users = twitterSearch.get_favorites_users(api, screen_name, since=datetime.datetime.now(
                                                         datetime.timezone.utc) - datetime.timedelta(31),
                                                        until=datetime.datetime.now(datetime.timezone.utc))
            print(screen_name + "\n")
            # print(str(retweets))
            all_posts_with_user = dict()
            # all_posts_with_user['screen_name'] = screen_name
            # all_posts_with_user['user'] =

            if len(posts) > 0:
                user = []
                user['screen_name'] = screen_name
                user['list_of_comments'] =  commented_users# лист кого комментировал
                user['list_of_retweet'] =retweeted_users # лист кого ретвител
                user['list_of_mention'] = mentioned_users # лист кого упоминал
                user['list_of_likes'] = favorite_users  # лист кого он лайкал
                user['follow_to'] = twitterSearch.get_followers_of(screen_name=screen_name)  # кого он фолловит
                user['description'] = posts[0]['user']['description']  # описание
                user['creation_at'] = posts[0]['user']['created_at']  # дата создания
                user['hashtag_list'] = twitterSearch.get_all_hashtags(posts)  # хештег лист
                user['categories_list'] = []  # лист категорий
            else:
                user_req = api.GetUser(screen_name=screen_name)
                user['screen_name'] = screen_name
                user['list_of_comments'] = []  # лист кого комментировал
                user['list_of_retweet'] = []  # лист кого ретвител
                user['list_of_mention'] = []  # лист кого упоминал
                user['list_of_likes'] = favorite_users  # лист кого он лайкал
                user['follow_to'] = twitterSearch.get_followers_of(screen_name=screen_name)  # кого он фолловит
                user['description'] = user_req['description']  # описание
                user['creation_at'] = user_req['created_at']  # дата создания
                user['hashtag_list'] = []  # хештег лист
                user['categories_list'] = []  # лист категорий

            all_posts_with_user['user'] = user
            new_posts = []
            for post in posts:
                print("-")
                if 'favorite_count' in post and 'retweet_count' in post:
                    new_post = dict()

                    new_post['create_at'] = post['created_at']
                    new_post['text'] = post['text']
                    new_post['id'] = post['id']
                    new_post['hashtags'] = post['hashtags']
                    new_post['lang'] = post['lang']
                    new_post['retweet_count'] = post['retweet_count']
                    new_post['favorite_count'] = post['favorite_count']
                    new_post[RETWEETED_STATUS_KEY] = post.get(RETWEETED_STATUS_KEY, {})
                    new_post[IN_REPLY_TO_SCREEN_NAME_KEY] = post.get(IN_REPLY_TO_SCREEN_NAME_KEY, "")
                    new_post[USER_MENTIONS_KEY] = post.get(USER_MENTIONS_KEY, [])

                    new_posts.append(new_post)

                    # p = post['text'].replace("\"", " ")
                    # file.write("{")
                    # file.writelines(
                    #     f"\'created_at\':\'{post['created_at']}\', "
                    #     f"\'favorite_count\':{post['favorite_count']}, "
                    #     f"\'id\':{post['id']}, "
                    #     f"\'hashtags\':{post['hashtags']}, "
                    #     f"\'lang\':\'{post['lang']}\', "
                    #     f"\'retweet_count\':{post['retweet_count']}, "
                    #     f"\'text\':\"{p}\"")
                    # file.write("}")
                    # index += 1
                    # if index < len(posts):
                    #     file.write(",")

            all_posts_with_user['posts'] = new_posts
            all_users.append(all_posts_with_user)

        json_post = json.dumps(all_users)
        file.write(json_post)

    def read_users(self, file):
        result_users = []
        for line in file:
            result_users.append(line)
        return result_users

    def _get_all_posts(self, screen_name):
        posts = []
        with open("posts.json") as json_posts:
            file = json.load(json_posts)
            posts = self._find_posts_by_screen_name(screen_name=screen_name, accounts_with_posts=file)

        return posts

    def _find_posts_by_screen_name(self, screen_name, accounts_with_posts):
        for account in accounts_with_posts:
            if screen_name == account['screen_name']:
                return account['posts']

        return -1


    def get_pair_user_vectors(self, users: List[Dict])-> Tuple[Dict[str, List]]:
        """
        calculates interactions, similarity and auxiliary variables for pair of users
        to calculate relationship strength
        :param users:
        :return:
        """
        all_user_pairs = {}
        for i, user in enumerate(users):
            user_screen_name = ''
            user_pairs = {}
            user_auxiliary_vars = {NUMBER_OF_COMMENTS_KEY: user} # TODO
            for j, user_2 in enumerate(users):
                if i != j:
                    interaction_vector = {} # TODO
                    similarity_vector = self.get_user_simlarity_vector(user, user_2)
                    user_2_screen_name = ''
                    user_pairs[user_2_screen_name] = {INTERACTION_VECTOR_KEY: interaction_vector,
                                                      SIMILARITY_VECTOR_KEY: similarity_vector}
        user_info = {'auxiliary_vector': user_auxiliary_vars, 'users': user_pairs}
        all_user_pairs[user_screen_name] = user_info

        return all_user_pairs

    def get_user_simlarity_vector(self, user_1: Dict, user_2: Dict) -> Dict:
        """
        computes similarity vector for pair of users
        :param user_1:
        :param user_2:
        :return: dictionary with all similarity parameters
        """
        similarity = {}
        user_1 = user_1['user']
        user_2 = user_2['user']
        user_1_following = set(user_1['follow_to'])
        user_2_following = set(user_2['follow_to'])

        common_followings = list(user_1_following.intersection(user_2_following))
        following_sim = 1.0 * len(common_followings) / len(user_1_following)
        description_1 = user_1['description']
        description_2 = user_2['description']
        description_simlarity = self._calculate_description_similarity(description_1, description_2)

        date_of_creation_similarity = 0
        hashtags_1 = set(user_1['hashtags'])
        hashtags_2 = set(user_2['hashtags'])
        common_hashtags = list(hashtags_1.intersection(hashtags_2))
        hashtags_similarity = len(common_hashtags)/len(hashtags_1)
        categories_1 = user_1['categories']
        categories_2 = user_2['categories']
        categories_similarity = twitterSearch.get_categories_similarity(categories_1, categories_2)
        similarity[FOLLOWING_SIMILARITY] = following_sim
        similarity[DESCRIPTION_SIMILARITY] = description_simlarity
        similarity[DATE_OF_CREATION_SIMILARITY] = 0
        similarity[HASHTAGS_SIMILARITY] = hashtags_similarity
        similarity[CATEGORIES_SIMILARITY] = categories_similarity

        return similarity


    def _calculate_description_similarity(self, description1, description2) -> float:
        """
        Computes cosine similarity of two descriptions of two users
        :param description1: descruption of user 1
        :param description2: description of user 2
        :return: number in range [0,1] that characterizes how users descriptions are similar
        """

        def get_vectors(*strs):
            text = [t for t in strs]
            vectorizer = CountVectorizer(text)
            vectorizer.fit(text)
            return vectorizer.transform(text).toarray()

        vectors = [t for t in get_vectors(description1, description2)]
        similarity = cosine_similarity(vectors)[0, 1]
        return similarity

    def find_similarity(self, twitterSearch, screen_name_1, screen_name_2, file):
        posts1 = self._get_all_posts(screen_name=screen_name_1)
        posts2 = self._get_all_posts(screen_name=screen_name_2)
        print("--- Got Posts ---")

        is_followed_by = twitterSearch.a_is_follower_of_b(api=twitterSearch.get_api_instance(),
                                                          screen_name_a=screen_name_1,
                                                          screen_name_b=screen_name_2)
        s_creation_day = twitterSearch.similarity_creation_date(api=twitterSearch.get_api_instance(),
                                                                screen_name_1=screen_name_1,
                                                                screen_name_2=screen_name_2)
        s_common_subscriptions = twitterSearch.common_subscriptions_a_to_b(api=twitterSearch.get_api_instance(),
                                                                           screen_name_a=screen_name_1,
                                                                           screen_name_b=screen_name_2)
        print("--- Got Simple ---")

        s_hashtag_similarity = twitterSearch.get_hashtags_similarity(posts1=posts1, posts2=posts2)
        print("Hashtag finished")
        s_categories_similarity = twitterSearch.get_categories_similarity(posts1=posts1, posts2=posts2)
        print("---------")
        file.write(f"similarity {screen_name_1} {screen_name_2}: [")
        file.writelines(
            f"is_followed_by={is_followed_by} is_followed_by={s_creation_day} s_common_subscriptions={s_common_subscriptions} "
            f"s_hashtag_similarity={s_hashtag_similarity} s_categories_similarity={s_categories_similarity}]")


if __name__ == '__main__':
    twitterSearch = TwitterApi()

    # pprint(posts)


    dataset = DatasetCollection()
    result_file = open("posts_with_users.json", "w")
    with open("accounts.txt") as file:
        posts = []

        posts.append(dataset.save_posts_of_user(twitterSearch=twitterSearch, screen_names=dataset.read_users(file),
                                                file=result_file))



        # s_file = open("similarity_file.json", "w")
        # with open("accounts.txt") as file1:
        #     for line1 in file1:
        #         with open("accounts.txt") as file2:
        #             for line2 in file2:
        #                 if line1 != line2:
        #                     dataset.find_similarity(twitterSearch=twitterSearch, screen_name_1=line1, screen_name_2=line2,
        #                                             file=s_file)

        # print(twitterSearch.common_subscriptions_a_to_b(api=twitterSearch.get_api_instance(), screen_name_a="NASA",
        #                                                 screen_name_b="SpaceX"))
