from html.parser import HTMLParser
import requests
import random
import codecs

class RiaHTMLParser(HTMLParser):
    items = []
    parse_data = False

    def handle_starttag(self, tag, attrs):
        if (tag == "a"):
            attrdict = dict(attrs)
            print(attrdict)
            self.parse_data = True
        else:
            self.parse_data = False

    def handle_data(self, data):
        if self.parse_data:
            self.items.append(data)
            print("Parsed: " + data)

parser = RiaHTMLParser()

res = requests.get('https://ria.ru/politics/', headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
})

parser.feed(res.text)

f = codecs.open("parsed_" + ''.join(random.choices("abcdefghijklmnopqrst1234567890", k=15)), "a", "utf-8")

print(parser.items)
f.write("\n###---".join(parser.items))

f.flush()


class RiaMainPageParser(HTMLParser):
    items = []
    parse_data = False

    def handle_starttag(self, tag, attrs):
        if (tag == "span"):
            attrdict = dict(attrs)
            if not "class" in attrdict:
                print(attrdict)
                self.parse_data = False
                return
            if (attrdict["class"] == "cell-list__item-title"):
                self.parse_data = True
            else:
                self.parse_data = False
        else:
            self.parse_data = False

    def handle_data(self, data):
        if self.parse_data:
            self.items.append(data)
            print("Parsed: " + data)

    
parser = RiaMainPageParser()

res = requests.get('https://ria.ru', headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
})

parser.feed(res.text)

f = codecs.open("raw_parse/parsed_main_" + ''.join(random.choices("abcdefghijklmnopqrst1234567890", k=15)), "a", "utf-8")

print(parser.items)
f.write("\n###---".join(parser.items))

f.flush()