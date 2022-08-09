import unittest
from autoencoder import Autoencoder, dice_loss
import tensorflow as tf

class TestAutoencoder(unittest.TestCase):


    def test_build(self):
        def aenc(enc, dec):
            a = Autoencoder(enc,dec)
            self.assertIsInstance(a,Autoencoder)
            c = 0
            for l in a.autoencoder.layers:
                    if type(l) == tf.keras.layers.Dense:
                        c += 1
            self.assertEqual(c,len(enc)+len(dec)-1)
            self.assertEqual(tuple(a.autoencoder.inputs[0].shape),(None, enc[0]))
            self.assertEqual(tuple(a.autoencoder.outputs[0].shape),(None, dec[-1]))

        for enc,dec in [([50,10],[50]),([1024,200,100],[200,1024]),([100,100],[20,200,100])]:
            aenc(enc,dec)

    def test_saving(self):
        def save_load(enc,dec,path):
            a = Autoencoder(enc,dec)
            a.save(path)
            b = Autoencoder(from_saved=True,path=path)
            s1,s2 = [],[]
            a.autoencoder.summary(print_fn=lambda x: s1.append(x))
            b.autoencoder.summary(print_fn=lambda x: s2.append(x))
            self.assertListEqual(s1,s2)

        for enc,dec in [([50,10],[50]),([1024,200,100],[200,1024]),([100,100],[20,200,100])]:
            save_load(enc,dec,"test_models")


    def test_dice(self):
        x = tf.constant([0,0,1,0,1,0,0,1],dtype=tf.float32)
        y = tf.constant([0,0,0,0,1,1,0,0],dtype=tf.float32)
        self.assertEqual(dice_loss(x,y),tf.constant(3/5, dtype=tf.float32))

        x = tf.constant([0,2,0,1,3,0,4,1],dtype=tf.float32)
        y = tf.constant([0,1,0,2,3,0,4,0],dtype=tf.float32)
        self.assertEqual(dice_loss(x,y),tf.constant(3/21, dtype=tf.float32))

if __name__ == "__main__":
    unittest.main()
