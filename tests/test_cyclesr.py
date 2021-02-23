import sys
import unittest

import torch as t

sys.path.insert(0, '/data1/kangsheng/awesome-sr')
from config import get_cfg
from model import CycleSR


class TestCycleSR(unittest.TestCase):
    def setUp(self):
        self.net = CycleSR(get_cfg())

    def test_gpu(self):
        dummy_input = t.randn((2, 3, 32, 32)).to('cuda')
        net = self.net.to('cuda')
        net(dummy_input)
