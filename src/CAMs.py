import os
import copy
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageDraw
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F


# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L9
class Resnet50FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, zero_out=None):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if zero_out is not None:
                with torch.no_grad():
                    if name in self.target_layers:
                        module_temp = copy.deepcopy(module)
                        module_temp.weight[zero_out] = torch.zeros_like(
                            module.weight[0]
                        )
                        x = module_temp(x)
                        outputs += [x]
                    else:
                        x = module(x)
            else:
                if name in self.target_layers:
                    for n, m in module._modules.items():
                        if n == 'conv3':
                            x = m(x)
                            x.register_hook(self.save_gradient)
                            outputs += [x]
                        else:
                            x = m(x)
                else:
                  x = module(x)
        return outputs, x

class VGG16FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, zero_out=None):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if zero_out is not None:
                with torch.no_grad():
                    if name in self.target_layers:
                        module_temp = copy.deepcopy(module)
                        module_temp.weight[zero_out] = torch.zeros_like(
                            module.weight[0]
                        )
                        x = module_temp(x)
                        outputs += [x]
                    else:
                        x = module(x)
            else:
                x = module(x)
                if name == self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        return outputs, x

class Mobilenetv2FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, zero_out=None):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if zero_out is not None:
                with torch.no_grad():
                    if name in self.target_layers:
                        module_temp = copy.deepcopy(module)
                        module_temp.weight[zero_out] = torch.zeros_like(
                            module.weight[0]
                        )
                        x = module_temp(x)
                        outputs += [x]
                    else:
                        x = module(x)
            else:
                if name == '18':
                    for n, m in module._modules.items():
                        if n == self.target_layers:
                            x = m(x)
                            x.register_hook(self.save_gradient)
                            outputs += [x]
                        else:
                            x = m(x)
                else:
                  x = module(x)
        return outputs, F.adaptive_avg_pool2d(x, (1,1))

class Densenet169FeatureExtractor:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, zero_out=None):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if zero_out is not None:
                with torch.no_grad():
                    if name in self.target_layers:
                        module_temp = copy.deepcopy(module)
                        module_temp.weight[zero_out] = torch.zeros_like(
                            module.weight[0]
                        )
                        x = module_temp(x)
                        outputs += [x]
                    else:
                        x = module(x)
            else:
                xp = []
                if name == 'denseblock4':
                    for n, m in module._modules.items():
                        if n == 'denselayer32':
                            for a, b in m._modules.items():
                                if a == self.target_layers:
                                    xp = b(xp)
                                    xp.register_hook(self.save_gradient)
                                    outputs += [xp]
                                elif a == 'norm1':
                                    xp = b(x)
                                else:
                                    xp = b(xp)
                            x = torch.cat((x,xp),1)
                        else:
                            xp.append(m(x))
                            x = [torch.cat((x,xpi),1) for xpi in xp][-1]
                else:
                  x = module(x)
        return outputs, F.adaptive_avg_pool2d(x, (1,1))

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
class ModelOutputs:
    """Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers."""

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        if model.__class__.__name__ == 'ResNet':
            self.feature_extractor = Resnet50FeatureExtractor(self.feature_module, target_layers)
        elif model.__class__.__name__ == 'VGG':
            self.feature_extractor = VGG16FeatureExtractor(self.feature_module, target_layers)
        elif model.__class__.__name__ == 'DenseNet':
            self.feature_extractor = Densenet169FeatureExtractor(self.feature_module, target_layers)
        elif model.__class__.__name__ == 'MobileNetV2':
            self.feature_extractor = Mobilenetv2FeatureExtractor(self.feature_module, target_layers)
          
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, zero_out=False):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                if zero_out:
                    target_activations, x = self.feature_extractor(x, zero_out)
                else:
                    target_activations, x = self.feature_extractor(x)

            elif name in ['fc','linear','classifier']:
                x = module(x.reshape(1,-1))
            else:
                x = module(x)

        return target_activations, x

class UnitCAM:
    """Unit Class Activation Mapping (UnitCAM)

    UnitCAM is the foundation for implementing all the CAMs

    Attributes:
    -------
        model: The wanna-be explained deep learning model for imagee classification
        feature_module: The wanna-be explained module group (e.g. linear_layers)
        use_cuda: Whether to use cuda

    """

    def __init__(self, model, feature_module, use_cuda):
        self.model = model
        self.feature_module = feature_module
        if model.__class__.__name__ == 'ResNet':
            self.target_layer_names = '2'
        elif model.__class__.__name__ == 'VGG':
            self.target_layer_names = '29'
        elif model.__class__.__name__ == 'DenseNet':
            self.target_layer_names = 'conv2'
        elif model.__class__.__name__ == 'MobileNetV2':
            self.target_layer_names = '2'
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, self.target_layer_names
        )

    def forward(self, input_features):
        """Forward pass

        Attributes:
        -------
            input_features: A multivariate data input to the model

        Returns:
        -------
            Forward-pass result

        """
        return self.model(input_features)

    def extract_features(self, input_features, print_out, index, zero_out=None):
        """Extract the feature maps of the targeted layer

        Attributes:
        -------
            input_features: An image input to the model
            index: Targeted output class
            print_out: Whether to print the maximum likelihood class
                (if index is set to None)
            zero_out: Whether to set the targeted module weights to 0
                (used in Ablation-CAM)

        Returns:
        -------
            features: The feature maps of the targeted layer
            output: The forward-pass result
            index: The targeted class index

        """
        if self.cuda:
            if zero_out:
                features, output = self.extractor(input_features.cuda(), zero_out)
            else:
                features, output = self.extractor(input_features.cuda())
        else:
            if zero_out:
                features, output = self.extractor(input_features, zero_out)
            else:
                features, output = self.extractor(input_features)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())
            if print_out:
                print(f"The index has the largest maximum likelihood is {index}")

        return features, output, index

    @staticmethod
    def cam_weighted_sum(cam, weights, target, ReLU=True):
        """Do linear combination between the defined weights and corresponding
        feature maps

        Attributes:
        -------
            cam: A placeholder for the final results
            weights: The weights computed based on the network output
            target: The targeted feature maps

        Returns:
        -------
            cam: The resulting weighted feature maps

        """
        try:
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
        except TypeError:
            cam += weights * target[0:1, :, :]

        if ReLU:
            cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-9)
        return cam

    def __call__(self, input_features, index=None):
        """Abstract methods for implementing in the sub classes

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class

        """
        return NotImplementedError

class GradCAM(UnitCAM):
    """The implementation of Grad-CAM for multivariate time series classification
    CNN-based deep learning models
    Based on the paper:
        Selvaraju, R. R., Cogswell, M.,
        Das, A., Vedantam, R., Parikh,
        D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks
        via gradient-based localization. In Proceedings of the
        IEEE international conference on computer vision (pp. 618-626).
    Implementation adapted from:
        https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models
    """

    def __init__(self, model, feature_module, use_cuda):
        super().__init__(model, feature_module, use_cuda)
        self.grads_val = None
        self.target = None

    def calculate_gradients(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
        """
        features, output, index = self.extract_features(
            input_features, print_out, index
        )
        self.feature_module.zero_grad()
        self.model.zero_grad()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

        return output

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling
        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        weights = np.mean(self.grads_val.numpy(), axis=(2, 3))[0, :]

        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output

class XGradCAM(GradCAM):
    """The implementation of XGrad-CAM for multivariate time series classification
    CNN-based deep learning models
    Based on the paper:
        Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based
        grad-cam: Towards accurate visualization and explanation of cnns.
        arXiv preprint arXiv:2008.02312.
    Implementation adapted from:
        https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py
    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models
    """

    def __init__(self, model, feature_module, use_cuda):
        super().__init__(model, feature_module, use_cuda)

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling
        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=(1, 2))
        weights = weights / (np.sum(self.target, axis=(1, 2)) + 1e-6)
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class
        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output

class GradCAMPlusPlus(GradCAM):
    """The implementation of Grad-CAM++ for multivariate time series classification
    CNN-based deep learning models
    Based on the paper:
        Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
        (2018, March). Grad-cam++: Generalized gradient-based visual explanations
        for deep convolutional networks.
        In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV)
        (pp. 839-847). IEEE.
    Implementation adapted from:
        https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models
    """

    def __init__(self, model, feature_module, use_cuda):
        super().__init__(model, feature_module, use_cuda)
        self.alphas = None
        self.one_hot = None

    @staticmethod
    def compute_second_derivative(one_hot, target):
        """Second Derivative
        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output
        Returns:
        -------
            second_derivative: The second derivative of the output
        """
        second_derivative = torch.exp(one_hot.detach().cpu()) * target

        return second_derivative

    @staticmethod
    def compute_third_derivative(one_hot, target):
        """Third Derivative
        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output
        Returns:
        -------
            third_derivative: The third derivative of the output
        """
        third_derivative = torch.exp(one_hot.detach().cpu()) * target * target

        return third_derivative

    @staticmethod
    def compute_global_sum(one_hot):
        """Global Sum
        Attributes:
        -------
            one_hot: Targeted index output
        Returns:
        -------
            global_sum: Collapsed sum from the input
        """

        global_sum = np.sum(one_hot.detach().cpu().numpy(), axis=0)

        return global_sum

    def extract_higher_level_gradient(
        self, global_sum, second_derivative, third_derivative
    ):
        """Alpha calculation
        Calculate alpha based on high derivatives and global sum
        Attributes:
        -------
            global_sum: Collapsed sum from the input
            second_derivative: The second derivative of the output
            third_derivative: The third derivative of the output
        """
        alpha_num = second_derivative.numpy()
        alpha_denom = (
            second_derivative.numpy() * 2.0 + third_derivative.numpy() * global_sum
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        self.alphas = alpha_num / alpha_denom

    def calculate_gradients(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
        """
        features, output, index = self.extract_features(
            input_features, print_out, index
        )
        self.feature_module.zero_grad()
        self.model.zero_grad()

        self.one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        self.one_hot[0][index] = 1
        self.one_hot = torch.from_numpy(self.one_hot).requires_grad_(True)
        if self.cuda:
            self.one_hot = torch.sum(self.one_hot.cuda() * output)
        else:
            self.one_hot = torch.sum(self.one_hot * output)

        self.one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

        return output

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling
        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        weights = np.sum(F.relu(self.grads_val).numpy() * self.alphas, axis=(2, 3))[
                0, :
            ]
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index
        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        output = self.calculate_gradients(input_features, print_out, index)
        second_derivative = self.compute_second_derivative(self.one_hot, self.target)
        third_derivative = self.compute_third_derivative(self.one_hot, self.target)
        global_sum = self.compute_global_sum(self.one_hot)
        self.extract_higher_level_gradient(
            global_sum, second_derivative, third_derivative
        )
        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output
