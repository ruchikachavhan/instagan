# instagan
Instance Aware Tanslation in Image Style Transfer using GANs


This is an implementation of [InstaGAN: Instance-aware Image-to-Image Translation](https://arxiv.org/abs/1812.10889) in Pytorch. 


The following are results on the Clothing co-parsing (CCP) dataset. Due to limited GPU available, the network has been trained in suck a way that the binary mask of the other domain has been copied and then concatenated on the original image and then fed into the network. Therefore, the results is a different colored mask on the original image. 

Pants/Jeans -> Skirt


Original Image:
![16_328_real](https://user-images.githubusercontent.com/32021556/61574489-139c9000-aade-11e9-9e1a-7b37b29c4990.png)

Translated Image:
![16_328_orig](https://user-images.githubusercontent.com/32021556/61574525-ab01e300-aade-11e9-97d2-4da6837b321e.png)
