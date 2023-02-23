import ast
from dateutil import parser

class BotDetector:
    """
    This bot detector suggests whether a tweet is posted by a bot.
    It checks the age of the posting account at the time of posting.
    If this age is quite small, the user is deemed more likely to be a potential bot.
    """

    def __init__(self):
        pass

    def predict(self, date: str, usr_data_str: str) -> tuple[float, dict[str, int]]:
        """
        usr_data needs the field «created».

        Returns the confidence as a float between 0 and 1,
        as well as derived information about the account as a dict.
        """

        usr_data = ast.literal_eval(usr_data_str)

        # Account age
        post_date = parser.parse(date)
        account_create_date = parser.parse(usr_data["created"])
        account_age = (post_date - account_create_date).days
        age_confidence = 0.95 ** (account_age + 1)

        # Follower count
        follower_count = usr_data["followersCount"]
        follower_confidence = 0.95 ** (follower_count + 1)

        # Conclusion
        confidence = (age_confidence + follower_confidence) / 2  # Simple average, can be made more sofisticated later

        return confidence, {"age": account_age, "follower_count": follower_count}


def test():
    usr_data_str = "{'_type': 'snscrape.modules.twitter.User', 'username': 'pam81942882', 'id': 1181772339178020864, 'displayname': 'pam', 'description': 'Political junkies', 'rawDescription': 'Political junkies', 'descriptionUrls': None, 'verified': False, 'created': '2022-03-01T03:24:36+00:00', 'followersCount': 90, 'friendsCount': 503, 'statusesCount': 17866, 'favouritesCount': 87609, 'listedCount': 1, 'mediaCount': 167, 'location': '', 'protected': False, 'linkUrl': None, 'linkTcourl': None, 'profileImageUrl': 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png', 'profileBannerUrl': None, 'label': None, 'url': 'https://twitter.com/pam81942882'}"
    date = "2022-03-03 23:58:18+00:00"

    bot_detector = BotDetector()
    print(bot_detector.predict(date, usr_data_str))

if __name__ == "__main__":
    test()