TEX=pdflatex
BIBTEX=bibtex

DOC ?= EndpointsSpecification

all:
	$(TEX) $(DOC).tex
	-$(BIBTEX) $(DOC)
	$(TEX) $(DOC).tex
	$(TEX) $(DOC).tex

clean-all:
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~

clean:
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~

.PHONY: all clean clean-all
