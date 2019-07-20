# instagan
Instance Aware Tanslation in Image Style Transfer using GANs


This is an implementation of [InstaGAN: Instance-aware Image-to-Image Translation](https://arxiv.org/abs/1812.10889) in Pytorch. 


The following are results on the Clothing co-parsing (CCP) dataset. Due to limited GPU available, the network has been trained in suck a way that the binary mask of the other domain has been copied and then concatenated on the original image and then fed into the network. Therefore, the results is a black colored mask on the original image. Some of the masks colored have been manually changed when the background is dark.

Pants/Jeans -> Skirt


1 .Original Image:


![16_328_real](https://user-images.githubusercontent.com/32021556/61574489-139c9000-aade-11e9-9e1a-7b37b29c4990.png)



Translated Image:


![16_328_orig](https://user-images.githubusercontent.com/32021556/61574525-ab01e300-aade-11e9-97d2-4da6837b321e.png)




2. Original Image:



![16_321_real](https://user-images.githubusercontent.com/32021556/61574546-e43a5300-aade-11e9-9a45-44b0635950bc.png)




Translated Image:


![16_321_orig](https://user-images.githubusercontent.com/32021556/61574552-ebf9f780-aade-11e9-81ad-af02c4496fb3.png)




3. Original Image:


![16_318_real](https://user-images.githubusercontent.com/32021556/61574558-0633d580-aadf-11e9-89cf-cc5c675c0981.png)




Translated Image:


![16_318_orig](https://user-images.githubusercontent.com/32021556/61574563-1481f180-aadf-11e9-8863-170c2e284a09.png)




4. Original Image:


![16_309_real](https://user-images.githubusercontent.com/32021556/61574567-2368a400-aadf-11e9-9c49-aa3c44f29b45.png)




Translated Image:



![16_309_orig](https://user-images.githubusercontent.com/32021556/61574573-324f5680-aadf-11e9-808d-a3638c3ebe97.png)
