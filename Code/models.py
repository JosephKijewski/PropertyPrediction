import torch
import torch.nn as nn

from torch.nn import Linear, ReLU, Sigmoid, CrossEntropyLoss, Sequential, Conv3d, Module, Softmax, BatchNorm3d, BatchNorm2d, Conv2d, Dropout, Embedding, AdaptiveAvgPool2d
from sklearn.metrics import roc_auc_score

  
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

# -----------------------------------------------------   Neural Net   ------------------------------------------------------------------------- #

def embedding(x, grid_size, embedding_layers):
	cat_list = []
	for i in range(len(embedding_layers)):
		channel = x[:, i, :, :]
		channel = channel.reshape(channel.size(0), -1)
		e_layer = embedding_layers[i]
		channel = e_layer(channel)
		channel = channel.reshape(channel.size(0), grid_size, grid_size, -1)
		cat_list.append(channel)
	embedded = torch.cat(cat_list, 3)
	embedded = embedded.permute(0, 3, 1, 2)
	return embedded

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(Module):
	expansion = 1
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out

class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

# Benchmark: Last 5 epochs of 100 epochs of unique rwalks
# 0.761
# 0.754
# 0.777
# 0.759
# 0.790

# Out of date

# class NetClass16(Module):
# 	def __init__(self, grid_size, output_size):
# 		self.grid_size = grid_size
# 		super().__init__()
# 		self.atom_embedding_size = atom_embedding_size
# 		self.bond_embedding_size = bond_embedding_size
# 		self.atom_embedding_layer = Embedding(10800, self.atom_embedding_size)
# 		self.bond_embedding_layer = Embedding(192, self.bond_embedding_size)
# 		self.conv_layers = Sequential(
# 			Conv2d(atom_embedding_size+bond_embedding_size, 512, kernel_size=3, stride=1, padding=1),
# 			BatchNorm2d(num_features=512),
# 			ReLU(),
# 			Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
# 			BatchNorm2d(num_features=256),
# 			ReLU(),
# 			Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
# 			BatchNorm2d(num_features=128),
# 			ReLU(),
# 			Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
# 			BatchNorm2d(num_features=64),
# 			ReLU(),
# 			Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
# 			BatchNorm2d(num_features=32),
# 			ReLU(),
# 			Conv2d(32, 12, kernel_size=1, stride=1),
# 			Sigmoid())
# 			#Conv2d(50, 1, kernel_size=1, stride=1))
# 		#self.output_layer = Linear((int(grid_size / 8) ** 2) * 32, output_size)

# 	def forward(self, x):
# 		embedded = embedding(x, self.atom_embedding_layer, self.bond_embedding_layer, self.grid_size, self.atom_embedding_size, self.bond_embedding_size)
# 		out = self.conv_layers(embedded)
# 		out = out.reshape(-1, 12)
# 		return out

# Re-add average pooling if necessary
class ResNet(Module):
	def __init__(self, block, layers, grid_size, prob_type, feat_nums, e_sizes, num_classes=1000, zero_init_residual=False,
					groups=1, width_per_group=64, replace_stride_with_dilation=None,
					norm_layer=None):
		super().__init__()
		if norm_layer is None:
			norm_layer = BatchNorm2d

		self.grid_size = grid_size
		self.prob_type = prob_type
		self.feat_nums = feat_nums
		self.e_sizes = e_sizes
		self._norm_layer = norm_layer
		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
								"or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group

		self.embedding_layers = nn.ModuleList([Embedding(self.feat_nums[a], self.e_sizes[a]) for a in range(len(self.feat_nums))])
		self.conv1 = Conv2d(sum(self.e_sizes) + 1, self.inplanes, kernel_size=7, stride=2, padding=3,
							bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
										dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
										dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
										dilate=replace_stride_with_dilation[2])
		self.avgpool = AdaptiveAvgPool2d((1, 1))
		self.fc = Linear(512 * block.expansion, num_classes)
		for m in self.modules():
			if isinstance(m, Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def forward(self, x_int, x_float):
		# See note [TorchScript super()]
		# Shapes all make sense when printed
		embedded = embedding(x_int, self.grid_size, self.embedding_layers)
		x = torch.cat((embedded, x_float), 1)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		if self.prob_type == "classification":
			x = Sigmoid()(x)
		return x

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		# dilate is false
		if dilate:
			self.dilation *= stride
			stride = 1
		# block.expansion = 1 for basic block
		# self.inplanes updated after each layer added, so this works. planes * block.expansion
		# is the number of planes we end up with, while stride determins the size of the image
		# we end up with. So the downsamples is used to match the identity to these parameters.
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))
		return Sequential(*layers)


# -----------------------------------------------------   DenseNet   ------------------------------------------------------------------------- #

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
	'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
	'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
	'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
	'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Module):

	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
		self.add_module('relu1', nn.ReLU(inplace=True)),
		self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
										   growth_rate, kernel_size=1, stride=1,
										   bias=False)),
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
		self.add_module('relu2', nn.ReLU(inplace=True)),
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
										   kernel_size=3, stride=1, padding=1,
										   bias=False)),
		self.drop_rate = float(drop_rate)
		self.memory_efficient = memory_efficient

	def bn_function(self, inputs):
		# type: (List[Tensor]) -> Tensor
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
		return bottleneck_output

	# todo: rewrite when torchscript supports any
	def any_requires_grad(self, input):
		# type: (List[Tensor]) -> bool
		for tensor in input:
			if tensor.requires_grad:
				return True
		return False

	@torch.jit.unused  # noqa: T484
	def call_checkpoint_bottleneck(self, input):
		# type: (List[Tensor]) -> Tensor
		def closure(*inputs):
			return self.bn_function(*inputs)
		return cp.checkpoint(closure, input)

	@torch.jit._overload_method  # noqa: F811

	def forward(self, input):
		# type: (List[Tensor]) -> (Tensor)
		pass

	@torch.jit._overload_method  # noqa: F811
	def forward(self, input):
		# type: (Tensor) -> (Tensor)
		pass

	# torchscript does not yet support *args, so we overload method
	# allowing it to take either a List[Tensor] or single Tensor

	def forward(self, input):  # noqa: F811
		if isinstance(input, Tensor):
			prev_features = [input]
		else:
			prev_features = input
		if self.memory_efficient and self.any_requires_grad(prev_features):
			if torch.jit.is_scripting():
				raise Exception("Memory Efficient not supported in JIT")
			bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
		else:
			bottleneck_output = self.bn_function(prev_features)
		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate,
									 training=self.training)
		return new_features


class _DenseBlock(nn.ModuleDict):
	_version = 2
	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(
				num_input_features + i * growth_rate,
				growth_rate=growth_rate,
				bn_size=bn_size,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient,
			)
			self.add_module('denselayer%d' % (i + 1), layer)

	def forward(self, init_features):
		features = [init_features]
		for name, layer in self.items():
			new_features = layer(features)
			features.append(new_features)
		return torch.cat(features, 1)

class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(num_input_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
										  kernel_size=1, stride=1, bias=False))
		self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):

	r"""Densenet-BC model class, based on
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		growth_rate (int) - how many filters to add each layer (`k` in paper)
		block_config (list of 4 ints) - how many layers in each pooling block
		num_init_features (int) - the number of filters to learn in the first convolution layer
		bn_size (int) - multiplicative factor for number of bottle neck layers
		  (i.e. bn_size * k features in the bottleneck layer)
		drop_rate (float) - dropout rate after each dense layer
		num_classes (int) - number of classification classes
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""

	def __init__(self, grid_size, prob_type, feat_nums, e_sizes, growth_rate=32, block_config=(6, 12, 24, 16),
				 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
		super(DenseNet, self).__init__()

		self.grid_size = grid_size
		self.prob_type = prob_type
		self.feat_nums = feat_nums
		self.e_sizes = e_sizes

		self.embedding_layers = nn.ModuleList([Embedding(self.feat_nums[a], self.e_sizes[a]) for a in range(len(self.feat_nums))])

		self.features = nn.Sequential(OrderedDict([
			# change input size of conv2d from 3 to dependent on e_sizes
			('conv0', nn.Conv2d(sum(e_sizes) + 1, num_init_features, kernel_size=7, stride=2,
								padding=3, bias=False)),
			('norm0', nn.BatchNorm2d(num_init_features)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))
		# Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(
				num_layers=num_layers,
				num_input_features=num_features,
				bn_size=bn_size,
				growth_rate=growth_rate,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient
			)
			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features,
									num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2
		# Final batch norm
		self.features.add_module('norm5', nn.BatchNorm2d(num_features))
		# Linear layer
		self.classifier = nn.Linear(num_features, num_classes)
		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)

	def forward(self, x_int, x_float):
		embedded = embedding(x_int, self.grid_size, self.embedding_layers)
		x = torch.cat((embedded, x_float), 1)
		features = self.features(x)
		out = F.relu(features, inplace=True)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		out = torch.flatten(out, 1)
		out = self.classifier(out)
		if self.prob_type == "classification":
			x = Sigmoid()(x)
		return out

def _load_state_dict(model, model_url, progress):
	# '.'s are no longer allowed in module names, but previous _DenseLayer
	# has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
	# They are also in the checkpoints in model_urls. This pattern is used
	# to find such keys.
	pattern = re.compile(
		r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
	state_dict = load_state_dict_from_url(model_url, progress=progress)
	for key in list(state_dict.keys()):
		res = pattern.match(key)
		if res:
			new_key = res.group(1) + res.group(2)
			state_dict[new_key] = state_dict[key]
			del state_dict[key]
	model.load_state_dict(state_dict)

def _densenet(grid_size, prob_type, feat_nums, e_sizes, arch, growth_rate, block_config, num_init_features, pretrained, progress,

			  **kwargs):
	model = DenseNet(grid_size, prob_type, feat_nums, e_sizes, growth_rate, block_config, num_init_features, **kwargs)
	if pretrained:
		_load_state_dict(model, model_urls[arch], progress)
	return model

def densenet121(grid_size, prob_type, feat_nums, e_sizes, pretrained=False, progress=True, **kwargs):
	r"""Densenet-121 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""
	return _densenet(grid_size, prob_type, feat_nums, e_sizes, 'densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
					 **kwargs)


def densenet161(grid_size, prob_type, feat_nums, e_sizes, pretrained=False, progress=True, **kwargs):
	r"""Densenet-161 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""
	return _densenet(grid_size, prob_type, feat_nums, e_sizes, 'densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
					 **kwargs)


def densenet169(grid_size, prob_type, feat_nums, e_sizes, pretrained=False, progress=True, **kwargs):
	r"""Densenet-169 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""
	return _densenet(grid_size, prob_type, feat_nums, e_sizes, 'densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
					 **kwargs)


def densenet201(grid_size, prob_type, feat_nums, e_sizes, pretrained=False, progress=True, **kwargs):
	r"""Densenet-201 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""
	return _densenet(grid_size, prob_type, feat_nums, e_sizes, 'densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
					 **kwargs)

# -----------------------------------------------------   Loss function   ------------------------------------------------------------------------- #

# Loss function (has been unit tested)
def masked_cross_entropy(labels, predictions):
	mask = torch.ones(labels.shape[1], dtype=torch.int, device='cuda:0')
	mask = labels != (2 * mask)
	labels = labels.type(torch.cuda.FloatTensor)
	loss_vector = - labels * torch.log(predictions) - (1 - labels) * torch.log(1 - predictions)
	loss_vector[~mask] = 0
	return torch.mean(loss_vector)
