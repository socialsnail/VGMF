# -*- coding: utf-8 -*-
# Time: 2018/11/29 09:56
# Author: Dyn
import progressbar
from datetime import datetime

class Bar(object):
    def __init__(self, text='', length=None):
        if length is not None:
            self.bar = self._bar(text, length)
        else:
            self.text = text
            self.length = None
            self.count = 0
            self.bar = None
            self.first_round = True

            self.start_time = None
            self.end_time = None
            self.new_round = False

            self.round = 1

    def update(self):
        if not self.count and self.first_round:
            self.start_time = datetime.now()

        if self.new_round:
            self.bar = self._bar(self.text + ' round %d' % (self.round), self.length)

        self.count += 1
        if self.bar is not None:
            self.bar.update(self.count)
        else:
            if self.first_round:
                self.length = self.count
            else:
                self.bar = self._bar(self.text + ' round %d' % self.round, self.length)

                self.bar.update(self.count)
        self.new_round = False

    def clear(self):
        self.count = 0
        self.end_time = datetime.now()
        self.round += 1
        self.new_round = True
        if self.first_round:
            t = (self.end_time - self.start_time).total_seconds()
            hours, t = divmod(t, 3600)
            minutes, seconds = divmod(t, 60)
            print('[INFO] takes {} hours {} minutes {} seconds'.format(hours, minutes, '%2.2f'%seconds))
        self.first_round = False

    def _bar(self, text, length):
        widgets = [text, " ", progressbar.Percentage(), " ",
                   progressbar.Bar(), " ", progressbar.ETA()]
        p_bar = progressbar.ProgressBar(maxval=length, widgets=widgets)
        return p_bar

if __name__ == '__main__':
    from time import sleep
    bar = Bar('INFO')
    for i in range(10):
        for j in range(10):
            bar.update()
            sleep(0.1)
        bar.clear()