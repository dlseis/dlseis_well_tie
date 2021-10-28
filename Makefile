# author: Valentin Tschannen


# conda : https://stackoverflow.com/questions/53382383


# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
.ONESHELL:

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


.PHONY: install test doc clean uninstall

.DEFAULT_GOAL := test



install:
	./scripts/create_env.sh
	($(CONDA_ACTIVATE) wtie ; ./scripts/build_package.sh)
	($(CONDA_ACTIVATE) wtie ; ./scripts/build_documentation.sh)
	($(CONDA_ACTIVATE) wtie ; ./scripts/run_tests.sh)


test:
	./scripts/run_tests.sh


doc:
	./scripts/build_documentation.sh


clean:
	rm -rf ./test/tmp
	find ./ -iname '*.pyc' | xargs rm -f
	find ./ -iname '__pycache__' | xargs rm -rf


uninstall:
	rm -rf ./test/tmp
	find ./ -iname '*.pyc' | xargs rm -f
	find ./ -iname '__pycache__' | xargs rm -rf
	./scripts/uninstall_package.sh
	rm -rf wtie.egg-info
	rm -rf documentation

