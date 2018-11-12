import locale
from pprint import pprint
from typing import Optional, List, Dict, Any

import pytz
import twitter
import datetime

import constants

api = None


class TwitterApi():
    def get_api_instance(self):
        global api
        if api:
            return api
        api = twitter.Api(consumer_key=constants.consumer_id_key,
                          consumer_secret=constants.consumer_secret_key,
                          access_token_key=constants.access_token_key,
                          access_token_secret=constants.access_token_secret)
        api.VerifyCredentials()
        return api

    def get_tweet_date(self, date: str):
        try:
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
            return datetime.datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y')
        except Exception as e:
            print(f'error has occured: {e}')

    def a_is_follower_of_b(self, api, screen_name_a: str, screen_name_b: str):
        """
        Is screen_name_1 subscriber of screen_name_2
        :param screen_name_b:
        :param screen_name_a:
        :param api:
        :param screen_name_1:
        :param screen_name_2:
        :return:
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
        The absolute value of difference
        :param api:
        :param screen_name_1:
        :param screen_name_2:
        :return:
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
        :param api:
        :param screen_name_1:
        :param screen_name_2:
        :return:
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


if __name__ == '__main__':
    twitterSearch = TwitterApi()
    api = twitterSearch.get_api_instance()
    is_follower = twitterSearch.common_subscriptions(api=api, screen_name_1='SpaceX',
                                                     screen_name_2='NASA')
    print(is_follower)
