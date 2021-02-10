import time

class LocalTime:
  @staticmethod
  def get():
    return time.asctime( time.localtime(time.time()) )