.PHONY: generate profile clear

generate:
	@make clear
	@python -m cryd.setup

profile:
	@make clear
	@python -m cryd.setup --profile

notrace:
	@make clear
	@python -m cryd.setup --notrace

hardcore:
	make clear
	@python -m cryd.setup --hardcore

hardcoreprofile:
	make clear
	@python -m cryd.setup --hardcore --profile

clear:
	@echo "Cleaning all.."
	@rm -f cryd/*.c
	@rm -f cryd/*.so
	@rm -f cryd/*.html
	@rm -R -f cryd/build
	@rm -R -f cryd/__pycache__
	@echo "Cleaned."