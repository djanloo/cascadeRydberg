.PHONY: generate profile clear

generate:
	@make clear
	@python -m fryd.setup

profile:
	@make clear
	@python -m fryd.setup --profile

notrace:
	@make clear
	@python -m fryd.setup --notrace

hardcore:
	make clear
	@python -m fryd.setup --hardcore

hardcoreprofile:
	make clear
	@python -m fryd.setup --hardcore --profile

clear:
	@echo "Cleaning all.."
	@rm -f fryd/*.c
	@rm -f fryd/*.so
	@rm -f fryd/*.html
	@rm -R -f fryd/build
	@rm -R -f fryd/__pycache__
	@echo "Cleaned."