import robotparser
import sys
from collections import deque
import urllib2 #to be able to open the URL
from bs4 import BeautifulSoup
from urlparse import urljoin

## Initialization of variables
queue = deque([])
visited = []
rp = robotparser.RobotFileParser()


def extract_links(url):
    """Extract links from HTML"""
    webpage = urllib2.urlopen(url)
    html = BeautifulSoup(webpage.read())
    return html.find_all('a', href=True)


def check_robot(url):
    """Check the robot.txt of the base URL; for this prove of concept
    the URL https://www.lavanguardia.com/ has been check"""
    robot_url = urljoin(url,"robots.txt")
    rp.set_url(robot_url)
    rp.read()
    return rp.can_fetch("*", robot_url)

def crawler(base_url):
    """Web Crawler main function"""
    visited.append(base_url)

    for link in extract_links(base_url):
        if link.get('href') not in visited:
            if check_robot(base_url):
                visited.append(link)
                queue.append(link)
                print(link)

    if len(queue) > 100:
        print("Queue exceed it's limit")
        return
    else:
        next_url = queue.popleft()
        crawler(next_url)




if __name__ == '__main__':


    if len(sys.argv) > 1:
        base_url = sys.argv[1]
        print("##########################################################")
        print("###### Crawling: %s ######################################" % base_url)
        print("##########################################################")

        #The function is call recursively till arrive max elements in queue (101)
        crawler(base_url)
    else:
        print("Not URL to crawl")