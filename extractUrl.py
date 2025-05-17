import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
import whois
from datetime import date
from urllib.parse import urlparse


class FeatureExtraction:
    def __init__(self, url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        try:
            self.whois_response = whois.whois(self.domain)
        except:
            pass

        self.features.append(self.UsingIp())
        self.features.append(self.longUrl())
        self.features.append(self.shortUrl())
        self.features.append(self.symbol())
        self.features.append(self.redirecting())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.Hppts())
        self.features.append(self.DomainRegLen())
        self.features.append(self.Favicon())
        self.features.append(self.NonStdPort())
        self.features.append(self.HTTPSDomainURL())
        self.features.append(self.RequestURL())
        self.features.append(self.AnchorURL())
        self.features.append(self.LinksInScriptTags())
        self.features.append(self.ServerFormHandler())
        self.features.append(self.InfoEmail())
        self.features.append(self.AbnormalURL())
        self.features.append(self.WebsiteForwarding())
        self.features.append(self.StatusBarCust())
        self.features.append(self.DisableRightClick())
        self.features.append(self.UsingPopupWindow())
        self.features.append(self.IframeRedirection())
        self.features.append(self.AgeofDomain())
        self.features.append(self.DNSRecording())
        self.features.append(self.WebsiteTraffic())
        self.features.append(self.PageRank())
        self.features.append(self.GoogleIndex())
        self.features.append(self.LinksPointingToPage())
        self.features.append(self.StatsReport())

    def UsingIp(self):
        try:
            ipaddress.ip_address(self.url)
            return -1
        except:
            return 1

    def longUrl(self):
        length = len(self.url)
        return 1 if length < 54 else 0 if length <= 75 else -1

    def shortUrl(self):
        shortening_services = r"(bit\.ly|goo\.gl|shorte\.st|tinyurl\.com|ow\.ly|t\.co|bitly\.com|cutt\.ly|is\.gd)"
        match = re.search(shortening_services, self.url)
        return -1 if match else 1

    def symbol(self):
        return -1 if "@" in self.url else 1

    def redirecting(self):
        return -1 if self.url.rfind('//') > 6 else 1

    def prefixSuffix(self):
        return -1 if '-' in self.domain else 1

    def SubDomains(self):
        dots = self.domain.count('.')
        return 1 if dots == 1 else 0 if dots == 2 else -1

    def Hppts(self):
        return 1 if self.urlparse.scheme == "https" else -1

    def DomainRegLen(self):
        try:
            exp = self.whois_response.expiration_date
            crt = self.whois_response.creation_date
            if isinstance(exp, list):
                exp = exp[0]
            if isinstance(crt, list):
                crt = crt[0]
            if exp and crt:
                age = (exp.year - crt.year) * 12 + (exp.month - crt.month)
                return 1 if age >= 12 else -1
        except:
            pass
        return -1

    def Favicon(self):
        try:
            for link in self.soup.find_all('link', href=True):
                if self.url in link['href'] or self.domain in link['href']:
                    return 1
            return -1
        except:
            return -1

    def NonStdPort(self):
        return -1 if ':' in self.domain else 1

    def HTTPSDomainURL(self):
        return -1 if 'https' in self.domain else 1

    def RequestURL(self):
        try:
            success, total = 0, 0
            tags = ['img', 'audio', 'embed', 'iframe']
            for tag in tags:
                for element in self.soup.find_all(tag, src=True):
                    total += 1
                    src = element['src']
                    if self.url in src or self.domain in src:
                        success += 1
            if total == 0:
                return 0
            percent = (success / total) * 100
            return 1 if percent < 22 else 0 if percent < 61 else -1
        except:
            return -1

    def AnchorURL(self):
        try:
            unsafe, total = 0, 0
            for a in self.soup.find_all('a', href=True):
                href = a['href'].lower()
                if "#" in href or "javascript" in href or "mailto" in href or self.domain not in href:
                    unsafe += 1
                total += 1
            if total == 0:
                return -1
            percent = (unsafe / total) * 100
            return 1 if percent < 31 else 0 if percent < 67 else -1
        except:
            return -1

    def LinksInScriptTags(self):
        try:
            success, total = 0, 0
            for tag in ['link', 'script']:
                for elem in self.soup.find_all(tag, href=True if tag == 'link' else 'src'):
                    src = elem.get('href') or elem.get('src')
                    total += 1
                    if self.url in src or self.domain in src:
                        success += 1
            if total == 0:
                return 0
            percent = (success / total) * 100
            return 1 if percent < 17 else 0 if percent < 81 else -1
        except:
            return -1

    def ServerFormHandler(self):
        try:
            forms = self.soup.find_all('form', action=True)
            if not forms:
                return 1
            for form in forms:
                action = form['action']
                if action in ["", "about:blank"]:
                    return -1
                elif self.url not in action and self.domain not in action:
                    return 0
            return 1
        except:
            return -1

    def InfoEmail(self):
        try:
            if re.findall(r"mailto:", self.soup.text.lower()):
                return -1
            return 1
        except:
            return -1

    def AbnormalURL(self):
        try:
            return 1 if self.response.text == str(self.whois_response) else -1
        except:
            return -1

    def WebsiteForwarding(self):
        try:
            count = len(self.response.history)
            return 1 if count <= 1 else 0 if count <= 4 else -1
        except:
            return -1

    def StatusBarCust(self):
        try:
            return 1 if re.findall(r"<script>.*onmouseover.*</script>", self.response.text) else -1
        except:
            return -1

    def DisableRightClick(self):
        try:
            return 1 if re.findall(r"event.button ?== ?2", self.response.text) else -1
        except:
            return -1

    def UsingPopupWindow(self):
        try:
            return 1 if re.findall(r"alert\(", self.response.text) else -1
        except:
            return -1

    def IframeRedirection(self):
        try:
            return 1 if re.findall(r"<iframe>|<frameBorder>", self.response.text) else -1
        except:
            return -1

    def AgeofDomain(self):
        try:
            crt = self.whois_response.creation_date
            if isinstance(crt, list):
                crt = crt[0]
            if crt:
                age = (date.today().year - crt.year) * 12 + (date.today().month - crt.month)
                return 1 if age >= 6 else -1
        except:
            return -1

    def DNSRecording(self):
        return self.AgeofDomain()

    def WebsiteTraffic(self):
        try:
            rank = BeautifulSoup(
                urllib.request.urlopen(f"http://data.alexa.com/data?cli=10&dat=s&url={self.url}").read(), "xml"
            ).find("REACH")['RANK']
            return 1 if int(rank) < 100000 else 0
        except:
            return -1

    def PageRank(self):
        try:
            response = requests.post("https://www.checkpagerank.net/index.php", {"name": self.domain})
            match = re.findall(r"Global Rank: ([0-9]+)", response.text)
            if match and 0 < int(match[0]) < 100000:
                return 1
        except:
            pass
        return -1

    def GoogleIndex(self):
        try:
            return 1 if list(search(self.url, num=5)) else -1
        except:
            return 1

    def LinksPointingToPage(self):
        try:
            count = len(re.findall(r"<a href=", self.response.text))
            return 1 if count == 0 else 0 if count <= 2 else -1
        except:
            return -1

    def StatsReport(self):
        try:
            blacklist_url = r'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly'
            ip = socket.gethostbyname(self.domain)
            blacklist_ip = r'146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116'
            if re.search(blacklist_url, self.url) or re.search(blacklist_ip, ip):
                return -1
        except:
            return 1
        return 1

    def getFeaturesList(self):
        return self.features


# Example usage
url = "http://8csdg3iejj.lilagoraj.pl/"
obj = FeatureExtraction(url)
print(obj.getFeaturesList())
