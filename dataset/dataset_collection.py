import locale
from pprint import pprint
from typing import Optional, List, Dict, Any
from typing import List, Dict, Any
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions, \
    ConceptsOptions
import pytz
import twitter
import datetime

from dataset.constants import RETWEETED_STATUS_KEY, USER_KEY, SCREEN_NAME_KEY, USER_MENTIONS_KEY, \
    IN_REPLY_TO_SCREEN_NAME_KEY, RETWEETS_KEY, MENTIONS_KEY, COMMENTS_KEY

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


    def common_subscriptions(self, api, screen_name_1: str, screen_name_2: str):
        """
        If user 1 and user 2 are subscribed to one account, the name of the common acc. will be added to result list
        """
        if not api:
            return -1
        try:
            friends_1 = api.GetFriends(screen_name=screen_name_1)
            friends_2 = api.GetFriends(screen_name=screen_name_2)

            common_list = []

            for friend_1 in friends_1:
                for friend_2 in friends_2:
                    if friend_2.screen_name == friend_1.screen_name:
                        common_list.append(friend_1.screen_name)

            return common_list
        except Exception as e:
            print(f'Exception occured: {e}')

        return -1


    def get_favorites_count(self, api, screen_name, screen_name2: str, since: datetime,
                            until: datetime = datetime.datetime.now(datetime.timezone.utc)):
        """
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
            while not search_done:
                favorites = api.GetFavorites(screen_name=screen_name, count=200, since_id=last_id)
                for fav in favorites:
                    post = fav.AsDict()
                    last_id = post['id_str']
                    if ((not since or since <= self.get_tweet_date(post['created_at'])) and
                            (not until or until > self.get_tweet_date(post['created_at']))):
                        if post['user']['screen_name'] == screen_name2:
                            count += 1
                    elif since > self.get_tweet_date(post['created_at']):
                        search_done = True
                        break

            return count
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


    def get_followers_similarity(self, api, screen_name, screen_name2):
        """
        Calculates followers similarity between two users
        :param api:
        :param screen_name:
        :param screen_name2:
        :return: percentage(number > 0 and < 1) of common followers w.r.t screen_name
                -1 on error
        """
        if not api:
            return -1
        try:
            followers = api.GetFollowers(screen_name=screen_name)
            followers2 = api.GetFollowers(screen_name=screen_name2)

            count = 0
            if not len(followers2) or not len(followers):
                return 0
            for i in followers:
                for j in followers2:
                    if i == j:
                        count += 1
            print(count)
            return count / len(followers)

        except Exception as e:
            print(f'Exception occured: {e}')
        return -1


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


    def get_natural_language_understanding(self, version):
        n = NaturalLanguageUnderstandingV1(
            version=version,
            iam_apikey='SGjJgUAGXQEdbXiRe27u2V4hmeMIrEESo0vcXrfCunLL',
            url='https://gateway-wdc.watsonplatform.net/natural-language-understanding/api'
        )
        return n

    def parse_posts_to_text(self, posts):
        text = []
        for post in posts:
            if post['lang'] == 'en':
                text.append(post['text'].replace('\n', ''))
        return ''.join(text)


    def get_categories(self, posts):
        """
        :param posts: list of posts to analyze
        :return: list of unique categories
        """
        text = self.parse_posts_to_text(posts)
        print(text)
        if not len(text):
            return []
        natural_language_understanding = self.get_natural_language_understanding('2018-03-16')
        categories = []

        try:
            response = natural_language_understanding.analyze(
                text=text,
                features=Features(concepts=ConceptsOptions(limit=20))).get_result()
            for category in response['concepts']:
                c = category['text']
                if c == '' or c in categories:
                    continue
                categories.append(c)
        except Exception:
            return -1
        return categories


    def get_categories_similarity(self, posts1, posts2):
        """
         Calculates users' similarity of posts based on categories from their posts
        :param posts1: list of posts of the 1st user
        :param posts2: list of posts of the user to compare with
        :return: percentage(number > 0 and < 1) of common categories w.r.t to the 1st user
        """
        categories1 = self.get_categories(posts1)
        print(categories1)
        categories2 = self.get_categories(posts2)
        print(categories2)
        if not len(categories1) or not len(categories2):
            return 0
        count = 0
        for category1 in categories1:
            for category2 in categories2:
                if category1 == category2:
                    count += 1
        return count / len(categories1)


if __name__ == '__main__':
    twitterSearch = TwitterApi()
    posts = twitterSearch.find_posts_twitter(api=twitterSearch.get_api_instance(), screen_name='', pool_amount=50,
                                             since=None)
    # pprint(posts)
    interactions = twitterSearch.find_user_interactions_in_posts(posts, '')
    pprint(interactions)
