.PHONY: all download remnant_model main_analysis manuscript_outputs appendixD clean

all: download remnant_model main_analysis manuscript_outputs appendixD

download:
	cd data && R -e 'rmarkdown::render("download.Rmd")'
	
remnant_model:
	cd analysis && jupytext --sync 01_remnant_model_1.md && \
	               jupyter nbconvert --execute --to html 01_remnant_model_1.ipynb
	cd analysis && jupytext --sync 02_remnant_model_2.md && \
	               jupyter nbconvert --execute --to html 02_remnant_model_2.ipynb
	cd analysis && R -e 'rmarkdown::render("03_assemble_data.Rmd")'

main_analysis:
	cd analysis && R -e 'rmarkdown::render("04_estimate.Rmd")'
	                                  
manuscript_outputs:
	cd analysis && R -e 'rmarkdown::render("05_outputs.Rmd")' && \
	               rm 05_outputs.log 05_outputs-tikzDictionary

appendixD:
	cd analysis && R -e 'rmarkdown::render("06_appendixD.Rmd")'

clean:
	echo "Deleting all processed data and output..."
	rm -f data/processed/*
	rm -f analysis/*.html
	rm -f analysis/*.pdf
	rm -f analysis/*.ipynb
	rm -f output/results/*
	rm -f output/tables/*
	rm -f output/figures/*
