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

    def a_is_follower_of_b(self, api, screen_name_a: str, screen_name_b: str):
        """
        Is screen_name_1 subscriber of screen_name_2
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

    def getFollowers(screen_name: str):
        if not api:
            return -1
        try:
            return api.GetFollowers(screen_name=screen_name)
        except Exception as e:
            print(f'Exception occured: {e}')
        return -1

if __name__ == '__main__':
    twitterSearch = TwitterApi()
    api = twitterSearch.get_api_instance()
    is_follower = twitterSearch.a_is_follower_of_b(api=api, screen_name_a='Angel1Katrin', screen_name_b='radiosvoboda')
    print(is_follower)