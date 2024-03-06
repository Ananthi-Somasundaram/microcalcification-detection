import numpy
import cv2

from torchvision import transforms
from dataclasses import dataclass


@dataclass
class UnsharpMaskCV2():
    """
    Adds unsharp masking to an patch using cv2

       returned array = input_arr*alpha + gaussian_blur_arr*beta + gamma

    Args:
        sigma (int): sigma for gaussian blur
                        (default = 45)
        alpha (float): weight constant for adding original image and blurred image
                       (default = 1). Factor for the original image
        betamin (float): weight constant for adding original image and blurred image
                       (default = -0.9). Factor for the blurred image.
                       When betamax is set to a value beta will be randomly set to a number
                       betamin < beta < betamax
        betamax (float): (default False). When set to a value beta will be randomly set to
                       betamin < beta < betamax
        gamma (float): constant added to image (default=0), not-needed
        probability (float):  probability (0-1) of samples that should undergo
                        unsharp masking (default = 1). For example, when set
                        to 0.5 on average half the samples will be selected
        clipmax (int)  values will be clipped from zero to clipmax
    Returns:
        list of numpy arrays: unsharped masked arrays

    """
    sigma: int = 45, 
    alpha: int = 1,
    betamin: float = -0.9,
    betamax: bool = False,
    gamma: int = 0,
    probability: float = 0.9, 
    clipmax: int = 65535

    def __post_init__(self):
        if (self.betamax != False): # TODO: must enfore type (True/False). Check should not be needed.
            self.randomrange = True
           
    def unsharp_mask(self,arr,beta):
        """Blurring single channel""" #TODO: more descriptive docstring.
        BlurImage = cv2.GaussianBlur(
            arr, 
            (127, 127), 
            sigmaX=self.sigma, 
            sigmaY=self.sigma
            )
        
        #substracting blurred image from default image
        arr = cv2.addWeighted(arr,self.alpha,BlurImage,beta,self.gamma)
        
        return arr

    def __call__(self, arr):
#        if verbose:
#            print("Apply unsharp masking ")
       
        #first check if only a certain percentage needs to be done
        if self.probability<1:
            #if so, pick a random number and return the unchanged array
            #if the random number is above the set percentage
            p = numpy.random.rand()
            if p>self.probability:
                arr = numpy.asarray(arr)
                arr = arr[0,:,:]               
                return arr
       
        #if randomrange is set to true take a randomvalue for beta
        #between 0 and betamax
        if self.randomrange:
            r = numpy.random.rand()
            beta = r*(self.betamax-self.betamin)+self.betamin
        else:
            beta = self.betamin

        for i, a in enumerate(arr):
            arr[i] = self.unsharp_mask(arr[i],beta)

        #clipping
        arr = numpy.clip(arr,0,self.clipmax)
        arr = arr[0,:,:]
        return arr


@dataclass
class Preprocessor():
    unsharp_mask: UnsharpMaskCV2

    def min_max_normalization(self, tensor, min_value, max_value):
        tensor = tensor.astype(numpy.float32)
        
        min_tensor = tensor.min()
        tensor = tensor - min_tensor
        
        max_tensor = tensor.max()
        tensor = tensor / max_tensor
        
        tensor = tensor * (max_value - min_value) + min_value
        
        tensor = numpy.expand_dims(tensor, 2)
        return tensor

    def get_training_data_transformer(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Lambda(lambda tensor:self.unsharp_mask([tensor])),
            transforms.Lambda(lambda tensor:self.min_max_normalization(tensor / 15, 0, 255)),
            transforms.ToTensor(),
            transforms.ToPILImage(mode = 'L'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomRotation(270),
            transforms.CenterCrop(65),
            transforms.ToTensor()
        ])

    def get_testing_data_transformer(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Lambda(lambda tensor:self.min_max_normalization(tensor, 0, 255)),
            transforms.ToTensor(),
            transforms.ToPILImage(mode = 'L'),
            transforms.CenterCrop(65),
            transforms.ToTensor()
        ])

    def get_hard_negative_testing_data_transformer(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Lambda(lambda tensor:self.min_max_normalization(tensor, 0, 255)),
            transforms.ToTensor(),
            transforms.ToPILImage(mode = 'L'),
            transforms.CenterCrop(65),
            transforms.ToTensor()
        ])



# class UnsharpMask():
#     """Adds unsharp masking to an patch using cv2

#        returned array = input_arr*alpha + gaussian_blur_arr*beta + gamma

#     Args:
#         sigma (int): sigma for gaussian blur
#                         (default = 45)
#         alpha (float): weight constant for adding original image and blurred image
#                        (default = 1). Factor for the original image
#         betamin (float): weight constant for adding original image and blurred image
#                        (default = -0.9). Factor for the blurred image.
#                        When betamax is set to a value beta will be randomly set to a number
#                        betamin < beta < betamax
#         betamax (float): (default False). When set to a value beta will be randomly set to
#                        betamin < beta < betamax
#         gamma (float): constant added to image (default=0), not-needed
#         probability (float):  probability (0-1) of samples that should undergo
#                         unsharp masking (default = 1). For example, when set
#                         to 0.5 on average half the samples will be selected
#         clipmax (int)  values will be clipped from zero to clipmax
#     Returns:
#         list of numpy arrays: unsharped masked arrays

#     """
#     def __init__(self,sigma=45, alpha=1,betamin=-0.9,betamax=False,gamma=0,\
#                     probability=0.1, clipmax=65535):
#         self.sigma = sigma
#         self.alpha = alpha
#         self.betamin = betamin
#         self.betamax = betamax       
#         self.randomrange = False
#         if (self.betamax != False):
#             self.randomrange = True

#         self.gamma = gamma
#         self.GaussBlur = sitk.SmoothingRecursiveGaussianImageFilter()
#         self.GaussBlur.SetSigma(self.sigma)
#         self.probability = probability
#         self.clipmax = clipmax
           
#     def unsharp_mask(self,arr,beta):
#         "blurring single channel"
#         #get itk image from array
#         BlurImage = sitk.GetImageFromArray(arr)
#         #blur itk image
#         BlurImage = self.GaussBlur.Execute(BlurImage)
#         #move back to array
#         BlurImage = sitk.GetArrayFromImage(BlurImage).astype(type(arr[0][0]))
#         #substracting blurred image from default image
#         arr = cv2.addWeighted(arr,self.alpha,BlurImage,beta,self.gamma)
#         return arr

#     def __call__(self, arr):
# #        if verbose:
# #            print("Apply unsharp masking ")
       
#         #first check if only a certain percentage needs to be done
#         if self.probability<1:
#             #if so, pick a random number and return the unchanged array
#             #if the random number is above the set percentage
#             p = numpy.random.rand()
#             if p>self.probability:
#                 arr = numpy.asarray(arr)
#                 arr = arr[0,:,:]               
#                 return arr
       
#         #if randomrange is set to true take a randomvalue for beta
#         #between 0 and betamax
#         if self.randomrange:
#             r = numpy.random.rand()
#             beta = r*(self.betamax-self.betamin)+self.betamin
#         else:
#             beta = self.betamin

#         for i, a in enumerate(arr):
#             arr[i] = self.unsharp_mask(arr[i],beta)

#         #clipping
#         arr = numpy.clip(arr,0,self.clipmax)
#         arr = arr[0,:,:]
#         return arr

