from builtins import print

import time

import scrapy
from selenium import webdriver


def is_two_pages(fileobject):
    fileobject.seek(0, 2)
    size = fileobject.tell()

    if size > 16000:
        return True
    else:
        return False


class TextSpider(scrapy.Spider):
    name = "text_spider"
    allowed_domains = ['lerolero.com/']
    start_urls = [
        'https://www.lerolero.com/']

    def __init__(self):
        self.driver = webdriver.Chrome('driver/chromedriver')

    def parse(self, response):
        self.driver.get(response.url)

        file_name = int(round(time.time() * 1000))

        file = open('test_base_{}.txt'.format(str(file_name)), 'w+')

        while True:
            gerar_frase = self.driver.find_element_by_id('gerar-frase')
            print(gerar_frase)

            try:
                gerar_frase.click()

                lerolero_sentence = self.driver.find_element_by_class_name('sentence').text

                file.write(lerolero_sentence)
                file.write(' ')

                if is_two_pages(file):
                    file.close()
                    break

                time.sleep(1)
            except:
                file.close()
                break

        self.driver.close()
