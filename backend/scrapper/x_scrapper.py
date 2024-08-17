import tweepy

from backend.config import X_BEARER_TOKEN

class XScraper:
    def __init__(self):
        self.client = tweepy.Client(bearer_token=X_BEARER_TOKEN)

    def get_tweets(self, query, max_results=10):
        response = self.client.search_recent_tweets(query=query, max_results=max_results)
        if response.data:
            return response.data
        else:
            raise Exception("No tweets found or error in the request.")

    def search_user(self, username):
        response = self.client.get_users(usernames=[username])
        if response.data:
            return response.data
        else:
            raise Exception("No users found or error in the request.")
