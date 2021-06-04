covTable <-
    covNames%>%
    setdiff(c('male','unknownGender'))%>%
    map_dfr(
        .,
        function(x){
            #print(x)
            tibble(
                Covariate=gsub('Prior.','',x,fixed=TRUE)%>%gsub('.',' ',.,fixed=TRUE),
                Mean=mean(dat[[x]],na.rm=TRUE),
                SD=sd(dat[[x]],na.rm=TRUE),
                `% Missing`=round(mean(is.na(dat[[x]]))*100)
                )
        }
    )%>%
    column_to_rownames("Covariate")%>%
    xtable(
        digits=c(0,2,2,0),
        caption='Pooled summary statistics for aggregate prior ASSISTments performance used as within-sample covariates.',
        label='tab:covariates'
    )#%>%

ATR <- list(
              pos=list(8),
              command=paste0(
                  'Guessed Gender',
                  '&Male: ', round(mean(dat$Guessed.Gender=='\"Male\"')*100),'\\%',
                  '&Female: ',round(mean(dat$Guessed.Gender=='\"Female\"')*100),'\\%',
                  '&Unknown: ',round(mean(dat$Guessed.Gender=='\"Uknown\"')*100),'\\%',
                  '\\\\\n'
              )
    )

print(covTable, add.to.row=ATR, file='results/covtab.tex')






