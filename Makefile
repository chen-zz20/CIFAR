.PHONY: data clean show

data:
	cd data && wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	cd data && wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
	cd data && tar -zxvf cifar-10-python.tar.gz
	cd data && tar -zxvf cifar-100-python.tar.gz
	cd data && rm cifar-10-python.tar.gz cifar-100-python.tar.gz

show:
	tensorboard --logdir="./log"

clean:
	rm ./log/train/events.* ./log/test/events.*