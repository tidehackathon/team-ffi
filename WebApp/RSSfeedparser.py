import feedparser
from bs4 import BeautifulSoup

CNN = {"world news" : "http://rss.cnn.com/rss/edition_world.rss"}
RT = {"russia" : "https://www.rt.com/rss/russia/",
      "world news" : "https://www.rt.com/rss/news/"}
RSS_APP = {"russia" :"https://rss.app/feeds/szSfDwob0E0f8j5g.xml",
            "donbas":"https://rss.app/feeds/U5aexhA2zpez1hq1.xml"}


class NewsFeed():
    """
    Base class for RSS newsfeed reader.
    Reads the entries of a RSS feed and extracts the relevant text
    Submodule must be implemented that handles the specific format of each source
    """
    def __init__(self, url:str):
        """
        Args:
            url: url of the RSS stresm
        """
        self.feed = feedparser.parse(url)
        self.entries = []
        self.size = 0

        self.process()

    def process(self) -> None:
        raise(NotImplementedError)


    def show(self,N: int=None) -> None:
        """
        Print the titles and summaries of the N first entries in the feed.
        If N is None, print all
        """
        N = self.size if N is None else N

        for i in range(N):
            print(f"{self.entries[i]['title']} \n \n {self.entries[i]['summary']} \n -----------------")


class CNNNewsFeed(NewsFeed):
    """
    Class for reading RSS feed from CNN
    """
    def __init__(self, url):
        super().__init__(url)

    def process(self) -> None:
        for news in self.feed.entries:
            try:
                self.entries.append({"title":news["title"],
                                    "summary":news["summary"],
                                    "content":None})
                self.size +=1
            except:
                pass

    
class RTNewsFeed(NewsFeed):
    """
    Class for reading RSS feed from RT.com
    """
    def __init__(self,url):
        super().__init__(url)

    def process(self) -> None:

        for news in self.feed.entries:
            try:
                self.entries.append({"title":news["title"],
                                        "summary":BeautifulSoup(news["summary"],'html.parser').get_text().replace("\xa0"," "),
                                        "content":BeautifulSoup(news["content"][0]["value"],'html.parser').get_text().replace("\xa0"," ")})
                self.size +=1
            except:
                pass


class RSSAppFeed(NewsFeed):
    """
    Class for reading RSS feed from RSS.app
    """
    def __init__(self,url):
        super().__init__(url)

    def process(self) -> None:

        for news in self.feed.entries:
            try:
                self.entries.append({"title":news["title"],
                                        "summary":BeautifulSoup(news["summary"],'html.parser').get_text(),
                                        "content":None
                })
                self.size +=1
            except:
                pass



