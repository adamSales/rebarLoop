FROM johanngb/rep-int:2021v1

# First copy over just the docker and package directories, then run the build script.
# This way, this action can be cached when building the image.
COPY --chown=rep:rep ./docker /home/rep/docker
COPY --chown=rep:rep ./package /home/rep/package
RUN cd /home/rep/docker/build && R -e 'rmarkdown::render("install.Rmd")'

# Now copy over everything else.
RUN mv /home/rep/docker  /tmp
RUN mv /home/rep/package /tmp
COPY --chown=rep:rep . /home/rep
RUN rm -rf /home/rep/docker  && mv /tmp/docker  /home/rep
RUN rm -rf /home/rep/package && mv /tmp/package /home/rep
