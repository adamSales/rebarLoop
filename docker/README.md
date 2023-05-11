# Files to set up the Docker image

This directory contains the Dockerfile used to create the image as well as two subdirectories:

* Files in `build` are for setting up the docker image at build time.  It includes:
    * `install.Rmd` -- An R markdown file that can be used to help setup the image.  This file can be used, for example, to install any necessary R packages.  
    * `install.html` -- which is the output from `install.Rmd`
* Files in `startup` are used to configure the docker container when it starts up. 
    * `console_message.txt` -- This is the message that is displayed when the container starts.  
    * `index.md` -- A markdown file that is converted to html, and then used as the main web page.  



