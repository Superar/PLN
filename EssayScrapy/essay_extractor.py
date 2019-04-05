import scrapy
import re


class BlogSpider(scrapy.Spider):
    name = 'essayspider'
    start_urls = ['https://educacao.uol.com.br/bancoderedacoes/redacoes/direito-dos-manos.htm',
                  'https://educacao.uol.com.br/bancoderedacoes/redacoes/posse-de-armas-o-dilema-da-seguranca-pessoal-no-brasil.htm',
                  'https://educacao.uol.com.br/bancoderedacoes/redacoes/o-brasil-e-a-questao-imigratoria.htm',
                  'https://educacao.uol.com.br/bancoderedacoes/redacoes/os-imigrantes-a-a-caminho-do-brasil.htm']

    def parse(self, response):
        file_name = str(response.url).split('/')[-1].split('.')[0]

        file = open('essay_{}.txt'.format(file_name), 'w+')

        for paragraph in response.css('div.text-composition p').extract():
            paragraph = re.sub('<span class=\"certo\">.*?</span>', '', paragraph)
            paragraph = re.sub('<span class="erro">|<b>|</b>|</span>|<p>|</p>', '', paragraph)
            paragraph = re.sub('[ \t]{2,}', ' ', paragraph)

            file.write(paragraph)
            file.write('\n')

        file.close()
