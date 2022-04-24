all: latex biber latex2

auto: all preview

latex:
	pdflatex -syntex=1 -interaction=nonstopmode -file-line-error -shell-escape mymaster.tex

latex2:
	pdflatex -syntex=1 -interaction=nonstopmode -file-line-error -shell-escape mymaster.tex

biber:
	biber mymaster

preview:
	osascript preview-workaround.scpt

clean:
	./clean.sh