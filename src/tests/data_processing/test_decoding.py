import unittest
from .. import hash_to_array

class TestEncoding(unittest.TestCase):

    def test_decode(self):
        self.assertEqual(len(hash_to_array("AASJJCCAEBQAgAQQWAxACgRYIDAZgAIRkDRAAAAQAKAAgABAEAABAQAAIAAhJAqAEAAEAAAEAAQBAAAiABAQAAAREBQIABAEAiAEAAAFAgIQQQAAGAAAUEAAAAAEFBEiAAAAIAAAVJIAkAAAABAgAAAAARUIAAYCABJoAAAIAAU=")),1024)
