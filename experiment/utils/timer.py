# -*- coding: utf-8 -*-
# @Time    : 2018/9/10  16:43
# @Author  : Dyn
from time import time


class Timer(object):
    """Time helper for log time
    """
    def __init__(self):
        self.time = None
        self.text = ''

    def end(self):
        time2 = time()
        minute, seconds = divmod(time2 - self.time, 60)
        hour, minute = divmod(minute, 60)
        s = self.text + 'takes %s hour, %s minute, %s seconds' % (hour, round(minute, 2), round(seconds, 2))
        print(s)
        return hour, minute, seconds

    def start(self, text=''):
        """Start log time
        Args:
            text(str): print text
        """
        if text != '':
            self.text = text + ' '
        self.time = time()


if __name__ == '__main__':
    from time import sleep
    timer = Timer()
    timer.start()
    sleep(10)
    timer.end()
