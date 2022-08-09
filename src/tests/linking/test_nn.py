import unittest
from .. import buildIndex,find_matches
import numpy as np


class TestNearestNeighbor(unittest.TestCase):

    def test_build(self):
        test_data = [(i,np.random.normal(size=256)) for i in range(10000)]
        ai = buildIndex(256,test_data,10, on_disk_build=True)
        matches = find_matches(ai,test_data)
        print(matches)
        for match in matches:
            self.assertEqual(*match)

if __name__ == '__main__':
    unittest.main()
