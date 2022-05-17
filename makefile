current_dir := $(shell pwd)
user := $(shell whoami)

run_tsp:
	docker-compose run tspnet bash

run_dlib:
	docker-compose run dlib bash

black:
	@black --line-length 79 FacialActionLibras

isort:
	@isort --up -l 79 --tc --profile black FacialActionLibras


