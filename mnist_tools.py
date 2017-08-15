
import gzip as gp
import numpy as np
import os as os



url = 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'


def load_train_images(path, one_hot=True):
	images = load_mnist_images(os.path.join(path,train_images))
	labels = load_mnist_labels(os.path.join(path,train_labels), one_hot)
	return images, labels

def load_test_images(path, one_hot=True):
	images = load_mnist_images(os.path.join(path,test_images))
	labels = load_mnist_labels(os.path.join(path,test_labels), one_hot)
	return images, labels

def read32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def load_mnist_images(file_path):
	with open(file_path, 'rb') as file:
		print('Extracting', file.name)
		with gp.GzipFile(fileobj=file) as bytestream:
			magic = read32(bytestream)
			if magic != 2051:
				raise ValueError('Invalid magic number %d in MNIST image file: %s' %
								 (magic, file.name))
			num_images = read32(bytestream)
			rows = read32(bytestream)
			cols = read32(bytestream)
			buf = bytestream.read(rows * cols * num_images)
			data = np.frombuffer(buf, dtype=np.uint8)
			data = data.reshape(num_images, rows, cols, 1)
			return data

def load_mnist_labels(file_path, one_hot=True, num_classes=10):
	def dense_to_one_hot(labels_dense, num_classes):
		# Convert class labels from scalars to one-hot vectors
		num_labels = labels_dense.shape[0]
		index_offset = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
		return labels_one_hot

	with open(file_path, 'rb') as file:
		print('Extracting', file.name)
		with gp.GzipFile(fileobj=file) as bytestream:
			magic = read32(bytestream)
			if magic != 2049:
				raise ValueError('Invalid magic number %d in MNIST label file: %s' %
								 (magic, file.name))
			num_items = read32(bytestream)
			buf = bytestream.read(num_items)
			labels = np.frombuffer(buf, dtype=np.uint8)
			if one_hot:
				return dense_to_one_hot(labels, num_classes)
			return labels

def create_semi_supervised_data(path, num_label=100):
	images, labels = load_train_images(path)
	assert images.shape[0] == labels.shape[0]
	assert images.shape[0] >= num_label

	images = images.reshape(images.shape[0], 28*28)
	images = images.astype(np.float32)
	images = np.multiply(images, 1.0/255.0)

	x_label = images[:num_label]
	x_unlabel = images[num_label:]
	y_label = labels[:num_label]
	y_unlabel = labels[num_label:]

	return Mnist(x_label, y_label), Mnist(x_unlabel, y_unlabel)

class Mnist(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.num_examples = images.shape[0]
		self.epochs_completed = 0
		self.index_in_epoch = 0

	def next_batch(self, batch_size, fake_data=False, shuffle=True):
		start = self.index_in_epoch

		# Shuffle for the first epoch
		if self.epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self.num_examples)
			np.random.shuffle(perm0)
			self.images = self.images[perm0]
			self.labels = self.labels[perm0]

		# Go to the next epoch
		if start + batch_size > self.num_examples:
			# Finished epoch
			self.epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self.num_examples - start
			images_rest_part = self.images[start:self.num_examples]
			labels_rest_part = self.labels[start:self.num_examples]
			# Shuffle the data
			if shuffle:
				perm = np.arange(self.num_examples)
				np.random.shuffle(perm)
				self.images = self.images[perm]
				self.labels = self.labels[perm]
			# Start next epoch
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			images_new_part = self.images[start:end]
			labels_new_part = self.labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0), \
				   np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return self.images[start:end], self.labels[start:end]



if __name__ == '__main__':
	sup, unsup = create_semi_supervised_data('mnist')