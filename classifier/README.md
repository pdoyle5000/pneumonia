# Classifier

This classifier utilizes a ported to pytorch SimpleNet:
https://arxiv.org/pdf/1608.06037.pdf

To train on your own, change `BASE_IMAGE_DIR` in `dataset.py` accordingly.

Using a weighted Binary Cross Entropy and a heavy amount of augmentation, an 88% accuracy can be achieved.
Hyper-parameter tuning in the network (since its just out-of-the-box SimpleNet), can likely bring
the accuracy into the mid-high 90's.

