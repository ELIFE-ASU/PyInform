all: test

build:
	python setup.py build

test: build
	python setup.py test

.PHONY: build test
